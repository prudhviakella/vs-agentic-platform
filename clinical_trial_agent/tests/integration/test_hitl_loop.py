"""
test_hitl_loop.py — HITL Pause/Resume Integration Tests
=========================================================
Tests the invoke_with_hitl() loop and handle_ask_user_input() handler.

No real LLM calls — agent.invoke() is mocked to return controlled
interrupt payloads and final responses.

What we're testing:
  1. Loop exits cleanly when no __interrupt__ present
  2. Loop handles one clarification interrupt correctly
  3. Loop handles multiple sequential clarification interrupts
  4. handle_ask_user_input produces the correct Command(resume=...) format
  5. Numeric option selection resolves to option text
  6. Freetext answer passes through unchanged
"""

import pytest
from unittest.mock import MagicMock, patch, call
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_interrupt_response(question: str, options: list, allow_freetext: bool = True) -> dict:
    """Build the __interrupt__ payload shape that HumanInTheLoopMiddleware produces."""
    return {
        "__interrupt__": [
            MagicMock(value={
                "action_requests": [{
                    "name": "ask_user_input",
                    "args": {
                        "question":       question,
                        "options":        options,
                        "allow_freetext": allow_freetext,
                        "user_answer":    "",
                    },
                    "description": "Input required\n\nTool: ask_user_input",
                }],
                "review_configs": [{
                    "action_name": "ask_user_input",
                    "allowed_decisions": ["approve", "edit", "reject"],
                }],
            })
        ],
        "messages": [HumanMessage(content="test")],
    }


def make_final_response(answer: str) -> dict:
    """Build a response with no __interrupt__ — agent finished."""
    return {
        "messages": [
            HumanMessage(content="test question"),
            AIMessage(content=answer),
        ]
    }


# ── handle_ask_user_input ──────────────────────────────────────────────────────

class TestHandleAskUserInput:

    def _get_action_request(self, question, options, allow_freetext=True):
        return {
            "name": "ask_user_input",
            "args": {
                "question":       question,
                "options":        options,
                "allow_freetext": allow_freetext,
                "user_answer":    "",
            }
        }

    def test_numeric_selection_resolves_to_option_text(self):
        from agent.run import handle_ask_user_input
        action_request = self._get_action_request(
            question="Which city?",
            options=["Delhi (DEL)", "Mumbai (BOM)", "Hyderabad (HYD)"],
        )
        with patch("builtins.input", return_value="3"):  # user types "3"
            cmd = handle_ask_user_input(action_request)

        assert isinstance(cmd, Command)
        edited = cmd.resume["decisions"][0]["edited_action"]["args"]
        assert edited["user_answer"] == "Hyderabad (HYD)"  # resolved from index

    def test_freetext_answer_passes_through(self):
        from agent.run import handle_ask_user_input
        action_request = self._get_action_request(
            question="Travel date?",
            options=["Apr 15", "Apr 20", "Apr 25"],
        )
        with patch("builtins.input", return_value="April 18th"):  # custom freetext
            cmd = handle_ask_user_input(action_request)

        edited = cmd.resume["decisions"][0]["edited_action"]["args"]
        assert edited["user_answer"] == "April 18th"

    def test_resume_format_is_edit_type(self):
        """
        Must always resume with type='edit' and edited_action pointing to ask_user_input.
        This is the exact format HumanInTheLoopMiddleware expects.
        """
        from agent.run import handle_ask_user_input
        action_request = self._get_action_request("Question?", ["A", "B"])
        with patch("builtins.input", return_value="1"):
            cmd = handle_ask_user_input(action_request)

        decision = cmd.resume["decisions"][0]
        assert decision["type"] == "edit"
        assert decision["edited_action"]["name"] == "ask_user_input"
        assert "user_answer" in decision["edited_action"]["args"]

    def test_out_of_range_number_treated_as_freetext(self):
        from agent.run import handle_ask_user_input
        action_request = self._get_action_request("Question?", ["A", "B"])
        with patch("builtins.input", return_value="99"):  # out of range
            cmd = handle_ask_user_input(action_request)

        edited = cmd.resume["decisions"][0]["edited_action"]["args"]
        assert edited["user_answer"] == "99"  # treated as freetext, not resolved


# ── invoke_with_hitl loop ──────────────────────────────────────────────────────

class TestInvokeWithHITLLoop:

    def test_returns_immediately_when_no_interrupt(self):
        """When agent has no __interrupt__, loop exits after one invoke()."""
        from agent.run import invoke_with_hitl

        agent    = MagicMock()
        final    = make_final_response("Metformin efficacy summary.")
        agent.invoke.return_value = final

        result = invoke_with_hitl(
            agent,
            messages=[{"role": "user", "content": "What is metformin efficacy?"}],
            config={"configurable": {"thread_id": "t1"}},
            context={"user_id": "test"},
        )

        assert agent.invoke.call_count == 1
        assert result["messages"][-1].content == "Metformin efficacy summary."

    def test_single_interrupt_then_final_answer(self):
        """
        One clarification interrupt → human answers → agent finishes.
        agent.invoke() should be called exactly twice.
        """
        from agent.run import invoke_with_hitl

        agent = MagicMock()
        agent.invoke.side_effect = [
            make_interrupt_response("Which city?", ["Delhi", "Mumbai", "Hyderabad"]),
            make_final_response("Flights from Hyderabad to Beijing."),
        ]

        with patch("builtins.input", return_value="3"):  # user picks Hyderabad
            result = invoke_with_hitl(
                agent,
                messages=[{"role": "user", "content": "Find flights India to China"}],
                config={"configurable": {"thread_id": "t2"}},
                context={"user_id": "test"},
            )

        assert agent.invoke.call_count == 2
        assert "Hyderabad" in result["messages"][-1].content

    def test_multiple_interrupts_loop_correctly(self):
        """
        Three sequential clarifications (city, date, cabin) → agent finishes.
        agent.invoke() should be called exactly four times.
        """
        from agent.run import invoke_with_hitl

        agent = MagicMock()
        agent.invoke.side_effect = [
            make_interrupt_response("Which city?",   ["Delhi", "Hyderabad"]),
            make_interrupt_response("Travel date?",  ["Apr 15", "Apr 20"]),
            make_interrupt_response("Cabin class?",  ["Economy", "Business"]),
            make_final_response("Flight options for Hyderabad to Beijing, April 15, Economy."),
        ]

        with patch("builtins.input", side_effect=["2", "1", "1"]):  # Hyderabad, Apr 15, Economy
            result = invoke_with_hitl(
                agent,
                messages=[{"role": "user", "content": "Find flights"}],
                config={"configurable": {"thread_id": "t3"}},
                context={"user_id": "test"},
            )

        assert agent.invoke.call_count == 4

    def test_second_invoke_receives_command_not_messages(self):
        """
        After an interrupt, the loop must pass a Command object to agent.invoke(),
        not the original messages dict. This is how LangGraph resumes the paused graph.
        """
        from agent.run import invoke_with_hitl

        agent = MagicMock()
        agent.invoke.side_effect = [
            make_interrupt_response("Question?", ["A", "B"]),
            make_final_response("Final answer."),
        ]

        with patch("builtins.input", return_value="1"):
            invoke_with_hitl(
                agent,
                messages=[{"role": "user", "content": "test"}],
                config={"configurable": {"thread_id": "t4"}},
                context={"user_id": "test"},
            )

        # First call: messages dict
        first_call_arg = agent.invoke.call_args_list[0][0][0]
        assert isinstance(first_call_arg, dict)
        assert "messages" in first_call_arg

        # Second call: Command object
        second_call_arg = agent.invoke.call_args_list[1][0][0]
        assert isinstance(second_call_arg, Command)
