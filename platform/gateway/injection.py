"""
gateway/injection.py — Prompt Injection Guard (Gateway Layer)
==============================================================
Runs domain-agnostic prompt injection detection at the gateway boundary
BEFORE the request reaches any agent. This is the first line of defence.

WHY at the gateway (not only in agent middleware):
  Agent middleware (ContentFilterMiddleware) catches domain-specific
  toxic patterns. Gateway injection check catches universal adversarial
  patterns that apply to ALL agents regardless of domain:
    "ignore previous instructions"
    "you are now..."
    "act as..."
  Blocking at the gateway means:
    - Zero agent invocation cost for blocked requests
    - One enforcement point for all current and future agents
    - Audit log entry before any agent state is touched

Uses check_prompt_injection() from vs-agent-core guardrails — the same
function the gateway would call regardless of which agent handles the request.

The agent's ContentFilterMiddleware still runs domain-specific checks
(toxic patterns, PII) after this gateway check passes — defence in depth.
"""

import logging

from fastapi import HTTPException

from core.aws import get_env

log = logging.getLogger(__name__)


def check_injection(text: str, request_id: str = "") -> None:
    """
    Run prompt injection detection on *text*.
    Raises HTTP 400 if an injection pattern is detected.

    WHY 400 (not 403):
      400 Bad Request signals that the input itself is malformed / invalid.
      403 Forbidden implies a valid request that is not authorised.
      A prompt injection attempt is malformed input, not an auth failure.

    Args:
        text:       The user's raw message content.
        request_id: Trace ID for log correlation.

    Raises:
        HTTPException 400 — injection pattern detected.
    """
    # Import here to keep the dependency on guardrails explicit
    # and to avoid circular imports at module load time.
    from agent.guardrails import check_prompt_injection

    is_clean, reason = check_prompt_injection(text)
    if not is_clean:
        log.warning(
            "[INJECTION] Blocked at gateway",
            extra={
                "request_id": request_id,
                "reason":     reason,
                "input_len":  len(text),
            },
        )
        raise HTTPException(
            status_code=400,
            detail=f"Request blocked: prompt injection pattern detected. {reason}",
        )

    log.debug(
        "[INJECTION] Passed",
        extra={"request_id": request_id, "input_len": len(text)},
    )
