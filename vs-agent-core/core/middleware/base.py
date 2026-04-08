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

        Priority:
          1. runtime.run_id     — set by LangGraph per invocation (most specific)
          2. runtime.thread_id  — stable across all hooks in the same conversation
          3. UUID fallback      — only if neither is available (should not happen
                                  in production; logged as a warning if it does)

        WHY NOT uuid fallback as primary:
          uuid.uuid4() generates a NEW id every call. If run_id is missing,
          before_agent and after_agent each get different UUIDs — the question
          stored in before_agent is never found in after_agent, so cache writes
          silently never happen.
        """
        run_id = getattr(runtime, "run_id", None)
        if run_id:
            return str(run_id)

        # thread_id is stable for the lifetime of a conversation — safe fallback
        thread_id = (
            getattr(runtime, "thread_id", None)
            or (getattr(runtime, "config", None) or {})
                .get("configurable", {})
                .get("thread_id")
        )
        if thread_id:
            return str(thread_id)

        # Last resort — log so this is visible during debugging
        import logging
        logging.getLogger(__name__).warning(
            "[BASE_MW] run_id and thread_id both missing from runtime — "
            "falling back to UUID. before_agent/after_agent bridges will NOT work."
        )
        return uuid.uuid4().hex[:8]