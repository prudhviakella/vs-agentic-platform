"""
base.py — BaseAgentMiddleware
==============================
Shared base class for all middleware in this package.

Extracts _get_run_id() which was duplicated across
SemanticCacheMiddleware, ActionGuardrailMiddleware, and TracerMiddleware.
"""

import uuid

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime


class BaseAgentMiddleware(AgentMiddleware):
    """
    Base class for all agent middleware in this project.
    Provides shared utilities used by multiple middleware classes.
    """

    def _get_run_id(self, runtime: Runtime) -> str:
        """
        Extract a stable identifier to bridge before_agent → after_agent.

        In LangChain 1.0, Runtime exposes only: context, store, stream_writer,
        merge, override, previous. There is no run_id or thread_id attribute.

        The thread_id is passed via context in router.py as session_id:
          context = {"user_id": ..., "session_id": body.thread_id, "domain": ...}

        Priority:
          1. runtime.context["session_id"]  — thread_id passed by the gateway
          2. runtime.context["user_id"]     — stable per user, less specific
          3. UUID fallback                   — should never happen in production
        """
        ctx = getattr(runtime, "context", None) or {}

        session_id = ctx.get("session_id")
        if session_id:
            return str(session_id)

        user_id = ctx.get("user_id")
        if user_id:
            return str(user_id)

        import logging
        logging.getLogger(__name__).warning(
            "[BASE_MW] session_id and user_id both missing from runtime.context — "
            "falling back to UUID. before_agent/after_agent bridges will NOT work. "
            "Ensure context is passed to agent.invoke() with session_id set."
        )
        return uuid.uuid4().hex[:8]