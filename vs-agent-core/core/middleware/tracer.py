"""
tracer.py — TracerMiddleware
==============================
Cross-cutting observability — FIRST in stack so even rejected requests are logged.

Core insight (slide 11):
  The state IS the trace. state["messages"] contains the complete execution
  record — every LLM call, every tool invocation, every result, in order.
  Walk that list → you have the full trace. No external service needed.

before_agent : record wall-clock start
after_agent  : extract trace from state, compute latency, log to LangSmith /
               MLflow / local, and persist to DynamoDB (fire-and-forget).

DynamoDB persistence
---------------------
All AWS/boto3 code lives in core.aws — this file has zero boto3 imports,
following the same pattern as every other module in the platform.
  core.aws.init_trace_table()  — creates table if absent, returns Table resource
  core.aws.put_trace()         — serialises and writes the trace item
  core.aws.get_trace_item()    — fetches a trace by run_id (admin/debug only)

Writes run in a background thread (fire-and-forget) so DynamoDB I/O never
adds latency to the agent response path. If the write fails, the error is
logged but the agent response is unaffected.
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from langchain.agents.middleware import AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

# One background worker is enough — traces are small and writes are fast.
# Capping at 1 prevents unbounded thread spawning under burst load.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracer-ddb")


class TracerMiddleware(BaseAgentMiddleware):
    """
    Cross-cutting observability — first in stack (before_agent) so every
    request is logged including rejected ones. Last on egress (after_agent)
    so the full execution trace is captured.

    Parameters
    ----------
    dynamodb_table_name : str | None
        DynamoDB table to persist traces into.
        If None, traces are kept in-memory only (dev / unit-test mode).
        Table is created automatically on first write if it does not exist.
    ttl_days : int
        Days before DynamoDB auto-expires a trace record (default: 30).
        Set to 0 to disable TTL.
    aws_region : str
        AWS region for the DynamoDB resource (default: "us-east-1").
    """

    def __init__(
        self,
        dynamodb_table_name: Optional[str] = None,
        ttl_days:            int            = 30,
        aws_region:          str            = "us-east-1",
    ):
        super().__init__()
        self._t0:     dict[str, float] = {}   # run_id -> wall-clock start
        self._traces: dict[str, dict]  = {}   # run_id -> trace dict (hot-path cache)

        self._table_name = dynamodb_table_name
        self._ttl_days   = ttl_days
        self._aws_region = aws_region

        # _ddb_table caches the boto3 Table resource after the first write.
        # None means "not yet initialised" (lazy) — not "DynamoDB disabled".
        self._ddb_table       = None
        self._ddb_table_ready = False
        self._ddb_table_lock  = threading.Lock()

        if not dynamodb_table_name:
            log.warning("[TRACER] No DynamoDB table configured — traces stored in-memory only.")

    # ─────────────────────────────────────────────────────── lifecycle hooks ──

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        run_id = self._get_run_id(runtime)
        self._t0[run_id] = time.time()
        log.info(f"[TRACER] T=0ms  request_received  run_id={run_id}")
        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        run_id  = self._get_run_id(runtime)
        elapsed = (time.time() - self._t0.pop(run_id, time.time())) * 1_000
        trace   = {"run_id": run_id, "elapsed_ms": round(elapsed, 2)}

        # ── Try LangSmith first ───────────────────────────────────────────
        try:
            from langsmith import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                trace.update({"run_tree_url": run_tree.url, "observability": "langsmith"})
                log.info(
                    f"[TRACER] T={elapsed:.0f}ms  langsmith_captured  "
                    f"run_id={run_id}  url={run_tree.url}"
                )
                self._traces[run_id] = trace
                self._persist_async(trace)
                return None
        except (ImportError, Exception):
            pass

        # ── Try MLflow next ───────────────────────────────────────────────
        try:
            import mlflow
            active_run = mlflow.get_active_run()
            if active_run:
                mlflow.log_metrics({"agent_latency_ms": elapsed})
                trace.update({"mlflow_run_id": active_run.info.run_id, "observability": "mlflow"})
                log.info(f"[TRACER] T={elapsed:.0f}ms  mlflow_captured  run_id={run_id}")
                self._traces[run_id] = trace
                self._persist_async(trace)
                return None
        except (ImportError, Exception):
            pass

        # ── Fallback: extract trace directly from LangGraph state ─────────
        # state["messages"] is the ground truth — no external service needed.
        trace.update(self._extract_from_state(run_id, elapsed, state.get("messages", [])))
        trace["observability"] = "local"
        log.info(
            f"[TRACER] T={elapsed:.0f}ms  run_complete  run_id={run_id}  "
            f"tools={trace.get('tools_called')}  llm_turns={trace.get('llm_turns')}"
        )
        self._traces[run_id] = trace
        self._persist_async(trace)
        return None

    # ──────────────────────────────────────────────── DynamoDB persistence ──

    def _get_table(self) -> Any | None:
        """
        Return the DynamoDB Table resource, initialising it on first call.

        Uses a lock so only one thread runs init_trace_table() even when
        multiple traces finish simultaneously at startup. After the first
        successful init, _ddb_table_ready=True skips the lock entirely.

        Returns None if DynamoDB is not configured (no table name given).
        """
        if not self._table_name:
            return None

        if self._ddb_table_ready:
            # Fast path — table already confirmed, no lock overhead.
            return self._ddb_table

        with self._ddb_table_lock:
            # Double-check inside the lock: another thread may have just finished.
            if not self._ddb_table_ready:
                from core.aws import init_trace_table
                self._ddb_table = init_trace_table(
                    table_name=self._table_name,
                    ttl_days=self._ttl_days,
                    region=self._aws_region,
                )
                self._ddb_table_ready = True

        return self._ddb_table

    def _persist_async(self, trace: dict) -> None:
        """
        Submit a DynamoDB write to the background thread pool.

        Fire-and-forget: the calling thread (agent response path) returns
        immediately. A failed write is logged but never propagated back —
        observability must never crash the happy path.
        """
        if not self._table_name:
            return  # DynamoDB not configured — skip silently

        # Shallow-copy so the background thread owns its reference and cannot
        # be affected by any mutation in the calling thread after this point.
        _executor.submit(self._write_trace, dict(trace))

    def _write_trace(self, trace: dict) -> None:
        """
        Background worker: resolve the table and delegate the write to core.aws.
        Runs in _executor thread — all exceptions are caught here.
        """
        try:
            table = self._get_table()
            if table is None:
                return
            from core.aws import put_trace
            put_trace(table=table, trace=trace, ttl_days=self._ttl_days)
        except Exception as exc:
            log.error(
                f"[TRACER] Background DynamoDB write failed  "
                f"run_id={trace.get('run_id')}  error={exc}"
            )

    # ──────────────────────────────────────────────────── static helpers ──

    @staticmethod
    def _extract_from_state(run_id: str, elapsed_ms: float, messages: list) -> dict:
        """
        Walk the LangGraph message list and produce a structured trace dict.
        Zero external dependencies, ~0 ms overhead.
        """
        question, answer, tool_calls, tool_results, llm_turns = "", "", [], [], 0

        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human" and not question:
                question = str(msg.content)
            elif isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "name": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                        })
                if msg.content:
                    llm_turns += 1
                    answer = str(msg.content)
            elif msg_type == "tool":
                tool_results.append({
                    "tool_name": getattr(msg, "name", "unknown"),
                    "content":   str(getattr(msg, "content", ""))[:200],
                    "is_error":  getattr(msg, "status", "") == "error",
                })

        return {
            "run_id":       run_id,
            "question":     question,
            "answer":       answer,
            "elapsed_ms":   round(elapsed_ms, 2),
            "tools_called": [tc["name"] for tc in tool_calls],
            "tool_count":   len(tool_calls),
            "tool_results": tool_results,
            "llm_turns":    llm_turns,
            "ts":           time.time(),
        }

    # ────────────────────────────────────────────────────────── public API ──

    def get_trace(self, run_id: str) -> Optional[dict]:
        """
        Return the in-memory trace for a given run_id.
        Gateway calls this to attach trace metadata to HTTP response headers.
        DynamoDB is the persistent store; this dict is the hot-path cache.
        """
        return self._traces.get(run_id)

    def get_trace_from_dynamodb(self, run_id: str) -> Optional[dict]:
        """
        Fetch a trace from DynamoDB by run_id.
        Admin / debug endpoints only — not called on the hot agent path.
        """
        if not self._table_name:
            log.warning("[TRACER] DynamoDB not configured — cannot fetch trace.")
            return None
        from core.aws import get_trace_item
        return get_trace_item(
            table_name=self._table_name,
            run_id=run_id,
            region=self._aws_region,
        )