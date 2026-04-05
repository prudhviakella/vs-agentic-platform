"""
hitl.py — ask_user_input Tool  (the single HITL gate)
=======================================================
This is the only tool the middleware watches (interrupt_on=True).
The LLM calls this whenever it needs clarification from the user.

All other tools auto-execute — not listed in interrupt_on, never cause a pause.

How it works:
  LLM calls ask_user_input(question, options) → middleware intercepts →
  graph PAUSES → human answers → Command(resume=edit+user_answer) →
  tool runs → returns user_answer as ToolMessage → LLM continues
"""

from typing import List
from langchain_core.tools import tool


@tool(parse_docstring=True)
def ask_user_input(
    question: str,
    options: List[str],
    allow_freetext: bool = True,
    user_answer: str = "",
) -> str:
    """
    Ask the user for a missing piece of information needed to complete the task.

    Call this once per missing parameter — ask ONE question at a time.
    Generate options that are SPECIFIC to the user's exact request context.
    Leave user_answer empty — it will be filled in by the human via the review UI.

    Args:
        question:       The specific clarifying question to ask the user.
        options:        Contextually relevant choices — derived from the user's
                        actual request, never generic placeholders like 'Option A'.
        allow_freetext: Whether the user can type a custom answer beyond the options.
        user_answer:    Leave empty. The human's answer is injected here on resume.

    Returns:
        The human's answer as a string. Use this value to complete the task.
    """
    return user_answer  # Human's answer flows back to LLM as a ToolMessage
