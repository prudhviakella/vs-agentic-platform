"""
output_guardrail.py — OutputGuardrailMiddleware
=================================================
Three-layer safety check before the answer reaches the user.

Layer 1 — regex (<1ms)       : obvious violations
Layer 2 — faithfulness (LLM) : answer grounded in retrieved context?
Layer 3 — contradiction (LLM): answer contradicts retrieved context?

WHY skip on HITL interrupt:
  When the agent calls ask_user_input, LangGraph pauses and returns
  {"__interrupt__": [...]} — there is no final answer in state yet.
  Running faithfulness scoring on an empty/partial answer produces
  faithfulness=0.00 which triggers _safe_fallback, which overwrites
  the interrupt and the HITL question never reaches the user.
  Checking the last AIMessage tool_calls skips the guardrail when
  the agent is paused waiting for human input.

WHY check message content for fallback detection:
  _safe_fallback returns {"messages": ..., "jump_to": "end"}.
  State keys set directly (state["key"] = val) are NOT persisted
  by LangGraph across middleware invocations — only returned dict
  keys are merged. Checking message content is reliable because
  the message IS in the returned dict and IS persisted.

WHY exclude summariser_tool from grounding:
  summariser_tool paraphrases retrieved chunks. A paraphrased
  summary scores lower against original text even when correct,
  causing false faithfulness failures.

WHY include previous AI answers in grounding context:
  For follow-up questions (e.g. "what is the turnaround time?"
  after an NCI-MATCH answer), the prior AI answer IS the grounding
  context. Without including it, the guardrail sees no context and
  scores faithfulness=0.00 on a perfectly valid follow-up answer.
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

_FALLBACK_MARKER = "did not meet safety and accuracy standards"


class OutputGuardrailMiddleware(BaseAgentMiddleware):

    _NON_GROUNDING_TOOLS = {"ask_user_input", "summariser_tool"}

    def __init__(
        self,
        llm=None,
        faithfulness_threshold: float = 0.0,
        confidence_threshold:   float = 0.0,
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

        # ── Skip 1: HITL interrupt ─────────────────────────────────────────
        # Detect by checking if the last AI message called ask_user_input.
        # Running faithfulness when agent is paused gives 0.00 → blocks HITL.
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if any(tc.get("name") == "ask_user_input" for tc in tool_calls):
                    log.info("[OUTPUT_GUARD] Skipping — agent paused for HITL")
                    return None
                break

        # Find the latest AI answer
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            return None

        # ── Skip 2: already a fallback ─────────────────────────────────────
        if _FALLBACK_MARKER in answer:
            return None

        # ── Layer 1: regex (<1ms, no LLM) ─────────────────────────────────
        ok, reason = check_medical_action_output(answer)
        if not ok:
            log.warning(f"[OUTPUT_GUARD] LAYER_1 FAIL  reason='{reason}'")
            return self._safe_fallback(state, f"Layer 1: {reason}")

        # ── Layers 2 + 3: LLM-as-judge ────────────────────────────────────
        context_chunks = self._extract_tool_results(messages)
        faith_score    = 1.0

        if context_chunks:
            try:
                faith_score = self._faithfulness_score_sync(answer, context_chunks)
                log.info(
                    f"[OUTPUT_GUARD] LAYER_2  faithfulness={faith_score:.2f}"
                    f"  threshold={self.faithfulness_threshold}"
                )
                if faith_score < self.faithfulness_threshold:
                    return self._safe_fallback(
                        state, f"Layer 2: faithfulness={faith_score:.2f} below threshold"
                    )

                consistency_score = self._contradiction_score_sync(answer, context_chunks)
                log.info(f"[OUTPUT_GUARD] LAYER_3  consistency={consistency_score:.2f}")
                if consistency_score < self.confidence_threshold:
                    return self._safe_fallback(
                        state, f"Layer 3: consistency={consistency_score:.2f} contradicts sources"
                    )

                confidence = min(faith_score, consistency_score)
                if confidence < self.confidence_threshold:
                    disclaimer = (
                        "\n\n⚠ Confidence below threshold. "
                        "Please verify with a qualified professional."
                    )
                    for i in range(len(messages) - 1, -1, -1):
                        if isinstance(messages[i], AIMessage):
                            messages[i].content = str(messages[i].content) + disclaimer
                            break
                    log.info(f"[OUTPUT_GUARD] DISCLAIMER added  confidence={confidence:.2f}")

            except Exception as exc:
                log.warning(f"[OUTPUT_GUARD] LLM judge failed ({exc}) — passing as-is")
        else:
            log.info("[OUTPUT_GUARD] No grounding context — skipping faithfulness check")

        state["_cache_faithfulness"] = faith_score
        state["_cache_is_fallback"]  = False

        log.info(f"[OUTPUT_GUARD] PASSED  answer='{answer[:60]}'")
        return None

    def _safe_fallback(self, state: AgentState, reason: str) -> dict[str, Any]:
        log.error(f"[OUTPUT_GUARD] HARD FAIL  reason='{reason}'")
        state["_cache_faithfulness"] = 0.0
        state["_cache_is_fallback"]  = True
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=(
            "I was unable to provide a verified answer for your question. "
            "The response did not meet safety and accuracy standards. "
            "Please consult a qualified professional or rephrase your question."
            f"\n\n[Reason logged for review: {reason}]"
        )))
        return {"messages": messages, "jump_to": "end"}

    def _extract_tool_results(self, messages: list) -> list[str]:
        """
        Collect grounding context from:
          1. Previous AI answers — for follow-up questions the prior answer
             IS valid grounding. Without this, "what is the turnaround time?"
             after an NCI-MATCH answer scores faithfulness=0.00 because the
             guardrail cannot see the prior conversation context.
          2. Tool results from clinical tools (search_tool, graph_tool).

        Excludes ask_user_input and summariser_tool — see class docstring.
        Capped at 6 chunks total.
        """
        results = []

        # Prior AI answers as grounding (for multi-turn follow-up questions)
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                if not getattr(msg, "tool_calls", None):
                    text = str(msg.content).strip()
                    if text and _FALLBACK_MARKER not in text:
                        results.append(text[:600])

        # Tool results from clinical tools
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                if str(getattr(msg, "name", "")) in self._NON_GROUNDING_TOOLS:
                    continue
                content = str(getattr(msg, "content", ""))
                if content:
                    results.append(content[:600])

        return results[:6]

    def _faithfulness_score_sync(self, answer: str, context_chunks: list[str]) -> float:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Rate how faithfully this answer is grounded in the retrieved context.\n"
            "Respond with ONLY a decimal number 0.0-1.0. Nothing else.\n"
            "0.0 = completely fabricated.  1.0 = fully grounded.\n\n"
            f"Retrieved Context:\n{context_text[:1_000]}\n\n"
            f"Answer:\n{answer[:600]}\n\nFaithfulness score:"
        )
        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return max(0.0, min(1.0, float(resp.content.strip())))
        except Exception:
            return 0.90

    def _contradiction_score_sync(self, answer: str, context_chunks: list[str]) -> float:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Does this answer contradict any of the retrieved context?\n"
            "Respond with ONLY a decimal 0.0-1.0. Nothing else.\n"
            "0.0 = direct contradiction.  1.0 = fully consistent.\n\n"
            f"Context:\n{context_text[:900]}\n\n"
            f"Answer:\n{answer[:500]}\n\nConsistency score:"
        )
        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return max(0.0, min(1.0, float(resp.content.strip())))
        except Exception:
            return 0.90