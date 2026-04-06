"""
gateway/middleware.py — FastAPI Middleware Stack
=================================================
ASGI middleware registered on the FastAPI app in main.py.
Runs on every request/response cycle, regardless of route.

Registered in order (outermost first):
  1. RequestContextMiddleware — assigns request_id, binds to log context
  2. TimingMiddleware         — measures and logs end-to-end latency
  3. CORSMiddleware           — cross-origin headers (FastAPI built-in)

WHY ASGI middleware (not FastAPI dependencies):
  Dependencies run per-route and can be skipped or overridden.
  ASGI middleware runs unconditionally on every request including
  404s, 422 validation errors, and unhandled exceptions — ensuring
  every request gets a request_id and timing even on error paths.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from vs_platform.observability.tracer import RequestContext

log = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique request_id to every incoming request.

    Behaviour:
      - If the caller provides X-Request-ID, reuse it (allows end-to-end
        tracing across upstream services that already assigned an ID).
      - Otherwise generate a fresh 12-char hex ID.
      - Bind to ContextVar so all log lines within the request include it.
      - Echo in X-Request-ID response header so clients can correlate logs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract agent name from URL path: /api/v1/{agent}/chat → {agent}
        parts = request.url.path.strip("/").split("/")
        agent = parts[2] if len(parts) >= 3 else ""

        ctx = RequestContext.from_request(
            request_id=request.headers.get("X-Request-ID"),
            agent=agent,
        )
        ctx.bind()

        try:
            response = await call_next(request)
        finally:
            ctx.unbind()

        response.headers["X-Request-ID"] = ctx.request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures wall-clock latency for every request and logs it.

    Emits a structured log line at the end of every request:
      {"msg": "Request complete", "method": "POST", "path": "/api/v1/...",
       "status": 200, "latency_ms": 312.4, "request_id": "..."}

    The latency_ms value is also added to the X-Latency-Ms response header
    for client-side performance monitoring.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = round((time.perf_counter() - t0) * 1_000, 2)

        log.info(
            "Request complete",
            extra={
                "method":     request.method,
                "path":       request.url.path,
                "status":     response.status_code,
                "latency_ms": elapsed,
            },
        )
        response.headers["X-Latency-Ms"] = str(elapsed)
        return response
