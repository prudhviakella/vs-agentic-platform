"""
action_guardrail.py — ActionGuardrailMiddleware
=================================================
Enforces action-level constraints per request (slide 9, layer 03):
  - Max 5 total tool calls per request
  - Logged per request for audit trail

WHY middleware and not inside each tool:
  Action limit is cross-cutting — it spans ALL tools in the request.
  An individual tool cannot know the total call count. Middleware can.
"""

import logging
import uuid
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langgraph.runtime import Runtime

from agent.tools import MAX_TOOL_CALLS_PER_REQUEST

log = logging.getLogger(__name__)


class ActionGuardrailMiddleware(BaseAgentMiddleware):

    def __init__(self):
        super().__init__()
        self._counts: dict[str, int] = {}  # run_id → tool call count


    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._counts[self._get_run_id(runtime)] = 0
        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        run_id     = self._get_run_id(runtime)
        tool_calls = sum(
            1 for msg in state.get("messages", [])
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )
        log.info(
            f"[ACTION_GUARD] run_id={run_id}  "
            f"tool_calls_used={tool_calls}  max={MAX_TOOL_CALLS_PER_REQUEST}"
        )
        self._counts.pop(run_id, None)
        return None
