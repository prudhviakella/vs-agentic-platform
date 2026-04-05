"""
observability/tracer.py — Request Tracing
==========================================
Generates and propagates a unique request_id for every API request.
The request_id is:
  - Injected into every log line via Python's logging context
  - Returned in the X-Request-ID response header
  - Passed into agent.invoke() so LangSmith traces are correlated

Usage (in FastAPI middleware):
  from platform.observability.tracer import RequestContext
  ctx = RequestContext.from_request(request)
  ctx.bind()          ← injects request_id into log context
  ctx.unbind()        ← cleans up after response
"""

import uuid
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ContextVar so request_id is isolated per async task / thread
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_agent_var:      ContextVar[str] = ContextVar("agent",      default="")


@dataclass
class RequestContext:
    """
    Holds per-request observability metadata.

    Attributes:
        request_id: Unique ID for this request — generated here or
                    forwarded from X-Request-ID header if provided by caller.
        agent:      Agent name derived from the URL path segment,
                    e.g. "clinical-trial" from /api/v1/clinical-trial/chat.
        user_id:    Extracted from the validated JWT claims.
    """
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent:      str = ""
    user_id:    str = "anonymous"

    @classmethod
    def from_request(
        cls,
        request_id: Optional[str] = None,
        agent:      str = "",
        user_id:    str = "anonymous",
    ) -> "RequestContext":
        """
        Build a RequestContext from FastAPI request data.

        If the caller provided X-Request-ID, reuse it for end-to-end
        tracing across services. Otherwise generate a fresh ID.
        """
        return cls(
            request_id=request_id or uuid.uuid4().hex[:12],
            agent=agent,
            user_id=user_id,
        )

    def bind(self) -> None:
        """
        Inject request_id and agent into the ContextVar storage so they
        are available to get_current_request_id() in log formatters and
        downstream code within the same async task.
        """
        _request_id_var.set(self.request_id)
        _agent_var.set(self.agent)
        log.debug(
            "Request started",
            extra={
                "request_id": self.request_id,
                "agent":      self.agent,
                "user_id":    self.user_id,
            },
        )

    def unbind(self) -> None:
        """Reset ContextVars after the response is sent."""
        _request_id_var.set("")
        _agent_var.set("")


def get_current_request_id() -> str:
    """Return the request_id for the current async task, or empty string."""
    return _request_id_var.get()


def get_current_agent() -> str:
    """Return the agent name for the current async task, or empty string."""
    return _agent_var.get()
