"""
gateway/rate_limiter.py — Per-User Rate Limiting
==================================================
Enforces request rate limits per user_id using an in-process sliding
window counter backed by a Python dict.

Limits:
  default:  60 requests / 60 seconds  (1 req/sec average)
  premium:  300 requests / 60 seconds
  service:  10,000 requests / 60 seconds (effectively unlimited)

WHY in-process (not Redis):
  In-process is zero-latency and zero-infrastructure for a single-instance
  deployment. For multi-instance deployments (ECS with >1 task), swap
  _WindowStore for a Redis-backed implementation — the interface stays the
  same.

WHY sliding window (not fixed window):
  Fixed windows allow 2x the limit at window boundaries. Sliding windows
  distribute traffic evenly and are fairer for sustained users.

Sliding window algorithm:
  Maintain a deque of request timestamps per user_id.
  On each request:
    1. Evict timestamps older than window_seconds from the left.
    2. If len(deque) >= limit → reject with 429.
    3. Append current timestamp → allow.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict

from fastapi import Depends, HTTPException

from vs_platform.gateway.auth import AuthContext, require_auth

log = logging.getLogger(__name__)


class _WindowStore:
    """
    In-process sliding window store.
    Safe for single-threaded async FastAPI (asyncio event loop is single-threaded).
    For multi-process deployments, replace with a Redis-backed implementation.
    """

    def __init__(self):
        self._windows: Dict[str, deque] = {}

    def check_and_record(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """
        Check if key is within the rate limit and record the attempt.
        Returns (allowed, remaining_requests).
        """
        now    = time.time()
        cutoff = now - window_seconds

        if key not in self._windows:
            self._windows[key] = deque()

        window = self._windows[key]

        # Evict timestamps outside the sliding window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= limit:
            return False, 0

        window.append(now)
        return True, limit - len(window)


_store = _WindowStore()


@dataclass
class RateLimit:
    limit:          int = 60
    window_seconds: int = 60


def _get_limit_for(auth: AuthContext) -> RateLimit:
    """
    Select the rate limit tier based on the auth context.
    Extend this to read per-tenant limits from DynamoDB when needed.
    """
    if auth.auth_mode == "api_key":
        return RateLimit(limit=10_000, window_seconds=60)  # internal callers
    if "premium" in auth.scopes:
        return RateLimit(limit=300, window_seconds=60)
    return RateLimit(limit=60, window_seconds=60)


async def check_rate_limit(auth: AuthContext = Depends(require_auth)) -> None:
    """
    FastAPI dependency — enforces per-user rate limit.
    Raises HTTP 429 with Retry-After header if the limit is exceeded.

    WHY Depends(require_auth) here (not just AuthContext):
      FastAPI cannot inject AuthContext automatically — it is a plain
      dataclass, not a FastAPI dependency. Declaring Depends(require_auth)
      tells FastAPI to resolve auth via the require_auth dependency first,
      then pass the result here. Without this, FastAPI would try to read
      AuthContext from the request body and fail.
    """
    rl  = _get_limit_for(auth)
    key = f"{auth.user_id}:{auth.tenant_id}"

    allowed, remaining = _store.check_and_record(key, rl.limit, rl.window_seconds)

    if not allowed:
        log.warning(
            "[RATE_LIMIT] Limit exceeded",
            extra={"user_id": auth.user_id, "tenant_id": auth.tenant_id},
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {rl.limit} requests per {rl.window_seconds}s.",
            headers={"Retry-After": str(rl.window_seconds)},
        )

    log.debug(
        "[RATE_LIMIT] Allowed",
        extra={
            "user_id":   auth.user_id,
            "remaining": remaining,
            "limit":     rl.limit,
        },
    )
