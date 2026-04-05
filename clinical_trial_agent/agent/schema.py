"""
schema.py — Runtime Context Schema
====================================
WHY TypedDict not dataclass:
  LangChain 1.0 docs explicitly state: "custom state schemas must be TypedDict
  types. Pydantic models and dataclasses are no longer supported."

Passed per-request via context= kwarg in invoke() / stream().
Accessible in middleware via runtime.context["user_id"] etc.
Accessible in @dynamic_prompt via request.runtime.context.
"""

from typing import TypedDict


class AgentContext(TypedDict, total=False):
    """
    Runtime configuration injected per-request.
    total=False means all keys are optional — safe default for any caller.
    """
    user_id:    str   # Who is asking
    session_id: str   # Which conversation session
    domain:     str   # "pharma" | "general" — controls cache threshold + domain framing
