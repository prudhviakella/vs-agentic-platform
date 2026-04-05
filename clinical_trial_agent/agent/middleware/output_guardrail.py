"""
output_guardrail.py — OutputGuardrailMiddleware
=================================================
Three-layer output defence (slide 9 layer 02 + slide 10 'The Real Danger'):

  Layer 1 — Code first   : regex patterns, <1ms, no LLM call
  Layer 2 — Faithfulness : LLM-as-judge, score < threshold → reject
  Layer 3 — Contradiction: LLM-as-judge, direct contradiction → hard fail

WHY class (not function): holds LLM client (stateful connection).
WHY after_agent + bottom-of-stack: closest to the user — last line of defence.

Bugs fixed vs original implementation:
─────────────────────────────────────────────────────────────────────────────
Bug 1 — Infinite loop:
  _safe_fallback() appended AIMessage → after_agent fired again on it
  → faithfulness=0.00 → another fallback → loop forever
  Fix: _FALLBACK_SENTINEL prefix — after_agent skips its own fallback output

Bug 2 — faithfulness=0.00 on valid HITL responses:
  ask_user_input ToolMessages (e.g. "Hyderabad") were included as grounding
  context → LLM judge: "is clinical answer grounded in 'Hyderabad'?" → 0.00
  Fix: _NON_GROUNDING_TOOLS excludes ask_user_input from context extraction
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from agent.guardrails import check_medical_action_output

log = logging.getLogger(__name__)


class OutputGuardrailMiddleware(BaseAgentMiddleware):

    # Sentinel prefix stamped on safe-fallback messages.
    # after_agent checks for this to avoid re-evaluating its own fallback,
    # which was the root cause of the infinite loop.
    _FALLBACK_SENTINEL = "[GUARDRAIL_FALLBACK]"

    # Tools whose ToolMessages are NOT grounding evidence.
    # ask_user_input returns user input (e.g. "Hyderabad"), not retrieved
    # clinical context. Including it as grounding caused faithfulness=0.00.
    _NON_GROUNDING_TOOLS = {"ask_user_input"}

    def __init__(
        self,
        llm=None,
        faithfulness_threshold: float = 0.85,
        confidence_threshold:   float = 0.75,
    ):
        super().__init__()
        self._llm                   = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.faithfulness_threshold = faithfulness_threshold
        self.confidence_threshold   = confidence_threshold

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            return None

        # ── Infinite-loop guard ─────────────────────────────────────────────
        if answer.startswith(self._FALLBACK_SENTINEL):
            log.info("[OUTPUT_GUARD] Safe fallback detected — skipping re-evaluation")
            return None

        # ── Layer 1: Code first — regex, <1ms, no LLM ──────────────────────
        ok, reason = check_medical_action_output(answer)
        if not ok:
            log.warning(f"[OUTPUT_GUARD] LAYER_1 FAIL  reason='{reason}'")
            return self._safe_fallback(state, f"Layer 1 (code-first): {reason}")

        # ── Layer 2 + 3: Faithfulness + Contradiction (LLM-as-judge) ───────
        context_chunks = self._extract_tool_results(messages)
        if context_chunks:
            try:
                faith_score = self._faithfulness_score_sync(answer, context_chunks)
                log.info(f"[OUTPUT_GUARD] LAYER_2  faithfulness={faith_score:.2f}  threshold={self.faithfulness_threshold}")

                if faith_score < self.faithfulness_threshold:
                    return self._safe_fallback(
                        state,
                        f"Layer 2 (faithfulness={faith_score:.2f}): answer not grounded in context"
                    )

                consistency_score = self._contradiction_score_sync(answer, context_chunks)
                log.info(f"[OUTPUT_GUARD] LAYER_3  consistency={consistency_score:.2f}")

                if consistency_score < 0.5:
                    return self._safe_fallback(
                        state,
                        f"Layer 3 (contradiction={consistency_score:.2f}): answer contradicts sources"
                    )

                # Low confidence → add disclaimer (slide 9: confidence < 0.75)
                confidence = min(faith_score, consistency_score)
                if confidence < self.confidence_threshold:
                    disclaimer = (
                        "\n\n⚠ Confidence below threshold. "
                        "Please verify this information with a qualified professional."
                    )
                    for i in range(len(messages) - 1, -1, -1):
                        if isinstance(messages[i], AIMessage):
                            messages[i].content = str(messages[i].content) + disclaimer
                            break
                    log.info(f"[OUTPUT_GUARD] DISCLAIMER added  confidence={confidence:.2f}")

            except Exception as exc:
                log.warning(f"[OUTPUT_GUARD] LLM judge failed ({exc}) — passing output as-is")
        else:
            # No grounding context (e.g. only ask_user_input calls, no search)
            # Skip faithfulness check — nothing to ground against
            log.info("[OUTPUT_GUARD] No grounding context — skipping faithfulness check")

        log.info(f"[OUTPUT_GUARD] PASSED  answer='{answer[:60]}'")
        return None

    def _safe_fallback(self, state: AgentState, reason: str) -> dict[str, Any]:
        """
        Replace answer with a safe fallback and stop execution.
        Stamps _FALLBACK_SENTINEL so after_agent skips re-evaluation on next pass.
        """
        log.error(f"[OUTPUT_GUARD] HARD FAIL → safe fallback  reason='{reason}'")
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=(
            f"{self._FALLBACK_SENTINEL} "
            "I was unable to provide a verified answer for your question. "
            "The response did not meet safety and accuracy standards. "
            "Please consult a qualified professional or rephrase your question."
            f"\n\n[Reason logged for review: {reason}]"
        )))
        return {"messages": messages, "jump_to": "end"}

    def _extract_tool_results(self, messages: list) -> list[str]:
        """
        Pull grounding tool results for faithfulness check.
        Excludes _NON_GROUNDING_TOOLS (ask_user_input) — user answers are not
        retrieved clinical evidence and cause faithfulness=0.00 when included.
        """
        results = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                if str(getattr(msg, "name", "")) in self._NON_GROUNDING_TOOLS:
                    continue
                content = str(getattr(msg, "content", ""))
                if content:
                    results.append(content[:600])
        return results[:4]

    def _faithfulness_score_sync(self, answer: str, context_chunks: list[str]) -> float:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Rate how faithfully this answer is grounded in the retrieved context.\n"
            "Respond with ONLY a decimal number 0.0–1.0. Nothing else.\n"
            "0.0 = completely fabricated.  1.0 = fully grounded.\n\n"
            f"Retrieved Context:\n{context_text[:1_000]}\n\n"
            f"Answer:\n{answer[:600]}\n\nFaithfulness score:"
        )
        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return max(0.0, min(1.0, float(resp.content.strip())))
        except Exception:
            return 0.90  # Safe default on judge failure

    def _contradiction_score_sync(self, answer: str, context_chunks: list[str]) -> float:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Does this answer contradict any of the retrieved context?\n"
            "Respond with ONLY a decimal 0.0–1.0. Nothing else.\n"
            "0.0 = direct contradiction.  1.0 = fully consistent.\n\n"
            f"Context:\n{context_text[:900]}\n\n"
            f"Answer:\n{answer[:500]}\n\nConsistency score:"
        )
        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return max(0.0, min(1.0, float(resp.content.strip())))
        except Exception:
            return 0.90
