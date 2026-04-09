"""
output_guardrail.py — OutputGuardrailMiddleware
=================================================
The last line of defence before the answer reaches the user.
Runs three checks in order -- cheapest first, most expensive last.

WHY do we need this?
  The LLM can hallucinate. In a clinical domain a hallucinated drug dose
  or contraindication could cause real harm.

  Example:
    LLM answers: "Metformin is safe for patients with eGFR below 15"
    Reality:     Metformin is CONTRAINDICATED below eGFR 30

    Layer 1 (regex)        -- may not catch this (no banned pattern)
    Layer 2 (faithfulness) -- catches it: answer not grounded in retrieved context
    Layer 3 (contradiction) -- catches it: answer directly contradicts source

THREE LAYERS -- cheapest to most expensive:

  Layer 1 -- Code-first regex (<1ms, no LLM call)
    Catches obvious violations: dosage claims, treatment recommendations.

  Layer 2 -- Faithfulness score (LLM-as-judge, gpt-4o-mini)
    "Is this answer grounded in what was actually retrieved?"
    Score 0.0-1.0. Below 0.85 -> reject.

  Layer 3 -- Contradiction score (LLM-as-judge, gpt-4o-mini)
    "Does this answer directly contradict the retrieved context?"
    Score 0.0-1.0. Below 0.50 -> hard fail.

WHY position 9 (bottom of stack)?
  after_agent runs in reverse order -- position 9 fires FIRST on egress.
  This means the guardrail is the first thing that runs after the agent
  and no other middleware can modify the answer after it is approved.

WHY class (not function)?
  Holds the LLM client as an instance variable -- one connection reused
  across all requests.
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

    # ask_user_input returns user answers like "Hyderabad" -- not clinical evidence.
    # Including these as grounding context gives faithfulness=0.00 on valid answers.
    # Only tool results from clinical tools (search, graph) are valid grounding.
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

        # Find the latest AI answer
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            return None

        # Layer 1 -- regex, no LLM, <1ms
        ok, reason = check_medical_action_output(answer)
        if not ok:
            log.warning(f"[OUTPUT_GUARD] LAYER_1 FAIL  reason='{reason}'")
            return self._safe_fallback(state, f"Layer 1: {reason}")

        # Layer 2 + 3 -- LLM-as-judge, only runs if tools were called
        context_chunks = self._extract_tool_results(messages)
        faith_score    = 1.0  # default when no context to evaluate against

        if context_chunks:
            try:
                # Layer 2 -- is the answer grounded in what was retrieved?
                faith_score = self._faithfulness_score_sync(answer, context_chunks)
                log.info(
                    f"[OUTPUT_GUARD] LAYER_2  faithfulness={faith_score:.2f}"
                    f"  threshold={self.faithfulness_threshold}"
                )
                if faith_score < self.faithfulness_threshold:
                    return self._safe_fallback(
                        state,
                        f"Layer 2: faithfulness={faith_score:.2f} below threshold"
                    )

                # Layer 3 -- does the answer contradict what was retrieved?
                consistency_score = self._contradiction_score_sync(answer, context_chunks)
                log.info(f"[OUTPUT_GUARD] LAYER_3  consistency={consistency_score:.2f}")
                if consistency_score < 0.5:
                    return self._safe_fallback(
                        state,
                        f"Layer 3: consistency={consistency_score:.2f} contradicts sources"
                    )

                # Passed both checks but confidence is low -- add disclaimer
                # instead of rejecting. User gets the answer with a warning.
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
                log.warning(f"[OUTPUT_GUARD] LLM judge failed ({exc}) -- passing output as-is")
        else:
            log.info("[OUTPUT_GUARD] No grounding context -- skipping faithfulness check")

        # Write signals to state for SemanticCacheMiddlewareWithRules.
        # This middleware is at position 9 so it fires first on egress --
        # these values are in state before SemanticCache (position 4) reads them.
        state["_cache_faithfulness"] = faith_score
        state["_cache_is_fallback"]  = False

        log.info(f"[OUTPUT_GUARD] PASSED  answer='{answer[:60]}'")
        return None

    def _safe_fallback(self, state: AgentState, reason: str) -> dict[str, Any]:
        """
        Replace the answer with a safe message and stop immediately.
        jump_to="end" ensures no other middleware can pass through a bad answer.
        """
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
        Collect grounding context from clinical tool results.
        Excludes ask_user_input -- user answers are not retrieved evidence.
        Capped at 4 chunks, 600 chars each to keep judge prompts short.
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
        """
        Ask gpt-4o-mini: is this answer grounded in the retrieved context?
        Returns 0.0 (fabricated) to 1.0 (fully grounded).
        Truncated to 1,000 chars of context + 600 chars of answer to control cost.
        """
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
        """
        Ask gpt-4o-mini: does this answer contradict the retrieved context?
        Returns 0.0 (direct contradiction) to 1.0 (fully consistent).

        Faithfulness catches answers that ignore the source.
        Contradiction catches answers that actively disagree with the source.
        Both checks are needed -- an answer can reference a source but misread it.
        """
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