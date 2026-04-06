"""
gateway/rate_limiter.py — Per-User Rate Limiting
==================================================
Enforces request rate limits per user_id using an in-process sliding
window counter backed by a Python dict.

Limits (configurable via SSM or defaults):
  default:      60 requests / 60 seconds  (1 req/sec average)
  premium:      300 requests / 60 seconds
  service:      unlimited (api_key auth_mode)

WHY in-process (not Redis):
  In-process is zero-latency and zero-infrastructure for a single-instance
  deployment. For multi-instance deployments (ECS with >1 task), swap
  _WindowStore for a Redis-backed implementation — the interface stays the
  same. The swap is a one-line change in build_rate_limiter().

WHY sliding window (not fixed window):
  Fixed windows allow 2x the limit at window boundaries (burst at the end
  of one window + start of the next). Sliding windows distribute traffic
  evenly and are fairer for sustained users.

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

from fastapi import HTTPException

from vs_platform.gateway.auth import AuthContext

log = logging.getLogger(__name__)


# ── Window store ───────────────────────────────────────────────────────────────

class _WindowStore:
    """
    In-process sliding window store.
    Thread-safe for single-threaded async FastAPI (asyncio event loop is single-threaded).
    For multi-process deployments, replace with RedisWindowStore.
    """

    def __init__(self):
        self._windows: Dict[str, deque] = {}

    def check_and_record(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """
        Check if *key* is within the rate limit and record the attempt.

        Returns:
            (allowed: bool, remaining: int)
              allowed   — True if the request is within the limit
              remaining — number of remaining requests in the current window
        """
        now = time.time()

        if key not in self._windows:
            self._windows[key] = deque()

        window = self._windows[key]

        # Evict timestamps outside the sliding window
        cutoff = now - window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= limit:
            remaining = 0
            return False, remaining

        window.append(now)
        remaining = limit - len(window)
        return True, remaining


_store = _WindowStore()


# ── Limit config ───────────────────────────────────────────────────────────────

@dataclass
class RateLimit:
    limit:          int = 60
    window_seconds: int = 60


def _get_limit_for(auth: AuthContext) -> RateLimit:
    """
    Select the rate limit tier based on the auth context.

    service accounts (api_key) — unlimited (internal callers are trusted).
    premium scope              — 300 req / 60s.
    default                    — 60 req / 60s.

    Extend this function to read per-tenant limits from DynamoDB or SSM
    when tenant-specific limits are required.
    """
    if auth.auth_mode == "api_key":
        return RateLimit(limit=10_000, window_seconds=60)  # effectively unlimited
    if "premium" in auth.scopes:
        return RateLimit(limit=300, window_seconds=60)
    return RateLimit(limit=60, window_seconds=60)


# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def check_rate_limit(auth: AuthContext) -> None:
    """
    FastAPI dependency — enforces per-user rate limit.
    Raises HTTP 429 with Retry-After header if the limit is exceeded.

    Inject after require_auth in route handlers:
      @router.post("/chat")
      async def chat(
          auth: AuthContext  = Depends(require_auth),
          _:    None         = Depends(check_rate_limit),   # ← add this
      ):
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
