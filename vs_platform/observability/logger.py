"""
observability/logger.py — Structured JSON Logger
==================================================
Configures structured JSON logging for CloudWatch / Datadog ingestion.
Every log line is a flat JSON object with consistent fields so log
aggregation queries work without regex parsing.

Log fields emitted on every record:
  ts          — ISO-8601 UTC timestamp
  level       — DEBUG | INFO | WARNING | ERROR | CRITICAL
  logger      — dotted module name (e.g. "vs_platform.gateway.auth")
  msg         — the log message string
  request_id  — injected automatically from ContextVar per request
  agent       — injected automatically from ContextVar when known

Usage:
  from vs_platform.observability.logger import get_logger
  log = get_logger(__name__)
  log.info("Auth passed", extra={"user_id": "abc"})
"""

import json
import logging
import sys
from datetime import datetime, timezone

from vs_platform.observability.tracer import get_current_request_id, get_current_agent


class _RequestContextFilter(logging.Filter):
    """
    Injects request_id and agent from ContextVar into every log record.

    WHY a Filter (not hardcoded in the Formatter):
      The filter runs before formatting and attaches values to the record
      object. This means extra={} fields passed by the caller are merged
      cleanly with the injected fields in the Formatter.

    WHY ContextVar (not threading.local):
      FastAPI uses asyncio -- multiple requests run on the same thread.
      threading.local would give the wrong request_id to concurrent requests.
      ContextVar is isolated per async task, which is exactly what we need.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Inject from ContextVar — empty string if no request is active
        if not getattr(record, "request_id", None):
            record.request_id = get_current_request_id()
        if not getattr(record, "agent", None):
            record.agent = get_current_agent()
        return True


class _JsonFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Extra fields passed via extra={} are merged into the top-level JSON.
    """

    # Standard LogRecord attributes that are not useful in the JSON output
    _SKIP = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs",
        "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Merge extra fields (includes request_id and agent from the filter)
        for key, val in record.__dict__.items():
            if key not in self._SKIP:
                payload[key] = val

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """
    Install the JSON formatter and request context filter on the root logger.
    Call once at application startup in main.py.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_RequestContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call at module level: log = get_logger(__name__)"""
    return logging.getLogger(name)
