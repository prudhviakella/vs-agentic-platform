"""
hitl.py — HumanInTheLoopMiddleware wrapper
============================================
Extends the built-in HumanInTheLoopMiddleware with a before_agent hook
that detects when ask_user_input has already been called and injects a
SystemMessage forcing the LLM to search immediately instead of asking
another clarifying question.

WHY this is needed:
  GPT-4o ignores prompt-level "ask only once" rules when it sees an
  ambiguous question and prior clarification in the message history.
  A deterministic SystemMessage injected into state is more reliable
  than a prompt instruction — the model sees it as fresh instruction
  right before the current turn.
"""

from typing import Any

from langchain.agents.middleware import HumanInTheLoopMiddleware, AgentState, hook_config
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime


class SingleClarificationHITLMiddleware(HumanInTheLoopMiddleware):
    """
    Drop-in replacement for HumanInTheLoopMiddleware.
    Identical interrupt behaviour — adds one guard:
    if ask_user_input was already called in this conversation,
    inject a SystemMessage telling the LLM to search immediately.
    """

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])

        # Check if ask_user_input tool result already exists in message history
        already_clarified = any(
            hasattr(msg, "type") and msg.type == "tool"
            and str(getattr(msg, "name", "")) == "ask_user_input"
            for msg in messages
        )

        if already_clarified:
            # LLM has already asked one question and received an answer.
            # Force it to search immediately instead of asking again.
            return {
                "messages": [
                    SystemMessage(
                        content=(
                            "INSTRUCTION: You have already asked one clarifying question "
                            "and received the user's answer. "
                            "You MUST NOT call ask_user_input again. "
                            "Call search_tool or graph_tool NOW to retrieve evidence "
                            "and answer the user's question."
                        )
                    )
                ]
            }

        return None