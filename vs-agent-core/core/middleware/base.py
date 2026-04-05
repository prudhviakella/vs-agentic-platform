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
        Extract a stable run identifier from the runtime context.
        Falls back to a fresh UUID if run_id is not present.
        """
        return str(getattr(runtime, "run_id", uuid.uuid4().hex[:8]))
