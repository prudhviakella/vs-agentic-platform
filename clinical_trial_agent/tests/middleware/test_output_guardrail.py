"""
test_output_guardrail.py — OutputGuardrailMiddleware Tests
===========================================================
Tests for the two bugs that caused the infinite loop.

Bug 1 — Infinite loop:
  _safe_fallback() appended AIMessage → after_agent fired again on it
  → faithfulness=0.00 → another fallback → loop forever
  Fix: _FALLBACK_SENTINEL prefix; after_agent skips its own fallback

Bug 2 — faithfulness=0.00 on valid HITL responses:
  ask_user_input ToolMessages ("Hyderabad") included as grounding context
  → "is clinical answer grounded in 'Hyderabad'?" → 0.00 → HARD FAIL
  Fix: _NON_GROUNDING_TOOLS excludes ask_user_input from context extraction

Fixtures (make_state, make_runtime) come from conftest.py — not redefined here.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.middleware.output_guardrail import OutputGuardrailMiddleware


def tool_message(content: str, name: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="call_001", name=name)


# ── Bug 1: Infinite loop guard ─────────────────────────────────────────────────

class TestFallbackSentinel:

    def test_skips_re_evaluation_of_own_fallback(self, make_state, make_runtime):
        mw    = OutputGuardrailMiddleware()
        state = make_state(
            HumanMessage(content="What is metformin?"),
            AIMessage(content=f"{OutputGuardrailMiddleware._FALLBACK_SENTINEL} I was unable to provide a verified answer."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result is None  # must not produce another fallback

    def test_safe_fallback_stamps_sentinel(self, make_state):
        mw     = OutputGuardrailMiddleware()
        state  = make_state(HumanMessage(content="test"))
        result = mw._safe_fallback(state, "test reason")
        assert result["messages"][-1].content.startswith(OutputGuardrailMiddleware._FALLBACK_SENTINEL)

    def test_normal_answer_goes_through_evaluation(self, make_state, make_runtime):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="0.95")
        mw    = OutputGuardrailMiddleware(llm=llm)
        state = make_state(
            HumanMessage(content="What is metformin?"),
            tool_message('{"results": ["Phase 3 RCT: 42% reduction"]}', "search_tool"),
            AIMessage(content="Metformin reduces HbA1c by 1.5% [Source: clinical_trials_db]."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result is None  # high faithfulness → passes


# ── Bug 2: Non-grounding tool exclusion ────────────────────────────────────────

class TestNonGroundingToolExclusion:

    def test_ask_user_input_excluded_from_grounding_context(self, make_state):
        mw   = OutputGuardrailMiddleware()
        msgs = [
            HumanMessage(content="Find flights"),
            tool_message("Hyderabad", "ask_user_input"),
            tool_message("2026-04-15", "ask_user_input"),
            AIMessage(content="Here are flights from Hyderabad to Beijing on April 15."),
        ]
        chunks = mw._extract_tool_results(msgs)
        assert len(chunks) == 0  # ask_user_input excluded

    def test_search_tool_included_in_grounding_context(self, make_state):
        mw   = OutputGuardrailMiddleware()
        msgs = [
            HumanMessage(content="What is metformin efficacy?"),
            tool_message('{"results": ["Phase 3 RCT: 42% reduction"]}', "search_tool"),
            AIMessage(content="Metformin shows 42% reduction vs placebo."),
        ]
        chunks = mw._extract_tool_results(msgs)
        assert len(chunks) == 1

    def test_mixed_tools_only_grounding_ones_included(self, make_state):
        mw   = OutputGuardrailMiddleware()
        msgs = [
            HumanMessage(content="What is the trial data?"),
            tool_message("Metformin Phase 3 RCT", "ask_user_input"),          # excluded
            tool_message('{"results": ["42% reduction"]}', "search_tool"),    # included
            tool_message('{"nodes": []}', "graph_tool"),                      # included
            AIMessage(content="The trial showed 42% reduction."),
        ]
        chunks = mw._extract_tool_results(msgs)
        assert len(chunks) == 2

    def test_no_grounding_context_skips_faithfulness_check(self, make_state, make_runtime):
        """
        When only ask_user_input ToolMessages exist, no grounding context is available.
        The faithfulness LLM must NOT be called — there is nothing to ground against.
        """
        llm = MagicMock()
        mw  = OutputGuardrailMiddleware(llm=llm)
        state = make_state(
            HumanMessage(content="Find flights"),
            tool_message("Hyderabad", "ask_user_input"),
            AIMessage(content="Here are flights from Hyderabad to Beijing."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result is None
        llm.invoke.assert_not_called()  # LLM judge must NOT have been called


# ── Layer 1: Code-first regex check ───────────────────────────────────────────

class TestLayer1CodeFirst:

    def test_blocks_medical_directive_without_calling_llm(self, make_state, make_runtime):
        llm = MagicMock()
        mw  = OutputGuardrailMiddleware(llm=llm)
        state = make_state(
            HumanMessage(content="What dose?"),
            AIMessage(content="You should take 500mg of metformin twice daily."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result is not None
        assert result.get("jump_to") == "end"
        llm.invoke.assert_not_called()  # regex caught it, no LLM needed

    def test_layer1_failure_stamps_sentinel(self, make_state, make_runtime):
        mw    = OutputGuardrailMiddleware()
        state = make_state(
            HumanMessage(content="dosage?"),
            AIMessage(content="Take 500 mg with food."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result["messages"][-1].content.startswith(OutputGuardrailMiddleware._FALLBACK_SENTINEL)


# ── Layer 2: Faithfulness threshold ───────────────────────────────────────────

class TestLayer2Faithfulness:

    def test_low_faithfulness_triggers_fallback(self, make_state, make_runtime):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="0.30")
        mw    = OutputGuardrailMiddleware(llm=llm)
        state = make_state(
            HumanMessage(content="What is metformin?"),
            tool_message('{"results": ["Phase 3 RCT data"]}', "search_tool"),
            AIMessage(content="Metformin cures cancer."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result is not None
        assert result.get("jump_to") == "end"

    def test_high_faithfulness_passes(self, make_state, make_runtime):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="0.92")
        mw    = OutputGuardrailMiddleware(llm=llm)
        state = make_state(
            HumanMessage(content="What is metformin efficacy?"),
            tool_message('{"results": ["42% reduction vs placebo"]}', "search_tool"),
            AIMessage(content="Metformin shows 42% reduction vs placebo [Source: clinical_trials_db]."),
        )
        result = mw.after_agent(state, make_runtime())
        assert result is None
