"""
observability/logger.py — Structured JSON Logger
==================================================
Configures structured JSON logging for CloudWatch / Datadog ingestion.
Every log line is a flat JSON object with consistent fields so log
aggregation queries work without regex parsing.

Log fields emitted on every record:
  ts          — ISO-8601 UTC timestamp
  level       — DEBUG | INFO | WARNING | ERROR | CRITICAL
  logger      — dotted module name (e.g. "platform.gateway.auth")
  msg         — the log message string
  request_id  — injected by RequestContextMiddleware per request
  agent       — injected when known (e.g. "clinical-trial")

Usage:
  from platform.observability.logger import get_logger
  log = get_logger(__name__)
  log.info("Auth passed", extra={"user_id": "abc", "agent": "clinical-trial"})
"""

import json
import logging
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Extra fields passed via extra={} are merged into the top-level JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }

        # Merge any extra fields the caller passed in extra={...}
        for key, val in record.__dict__.items():
            if key not in (
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName",
            ):
                payload[key] = val

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """
    Install the JSON formatter on the root logger.
    Call once at application startup in main.py.

    Args:
        level: Log level string — "DEBUG" | "INFO" | "WARNING" | "ERROR"
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call at module level: log = get_logger(__name__)"""
    return logging.getLogger(name)
