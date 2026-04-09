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

The agent's ContentFilterMiddleware still runs domain-specific checks
after this gateway check passes — defence in depth.
"""

import logging

from fastapi import HTTPException

log = logging.getLogger(__name__)


def check_injection(text: str, request_id: str = "") -> None:
    """
    Run prompt injection detection on text.
    Raises HTTP 400 if an injection pattern is detected.

    WHY 400 (not 403):
      400 Bad Request signals that the input itself is malformed.
      403 Forbidden implies a valid request that is not authorised.
      A prompt injection attempt is malformed input, not an auth failure.
    """
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
