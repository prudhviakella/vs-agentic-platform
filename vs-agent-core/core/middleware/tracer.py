"""
tracer.py — TracerMiddleware
==============================
Cross-cutting observability — FIRST in stack so even rejected requests are logged.

Core insight (slide 11):
  The state IS the trace. state["messages"] contains the complete execution
  record — every LLM call, every tool invocation, every result, in order.
  Walk that list → you have the full trace. No external service needed.

before_agent : record wall-clock start
after_agent  : extract trace from state, compute latency, log to LangSmith / MLflow / local
"""

import logging
import time
from typing import Any, Optional

from langchain.agents.middleware import AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)


class TracerMiddleware(BaseAgentMiddleware):
    """
    Cross-cutting observability — first in stack (before_agent) so every
    request is logged including rejected ones. Last on egress (after_agent)
    so the full execution trace is captured.
    """

    def __init__(self):
        super().__init__()
        self._t0:     dict[str, float] = {}
        self._traces: dict[str, dict]  = {}

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

        # LangSmith
        try:
            from langsmith import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                trace.update({"run_tree": run_tree, "observability": "langsmith"})
                log.info(f"[TRACER] T={elapsed:.0f}ms  langsmith_captured  run_id={run_id}  url={run_tree.url}")
                self._traces[run_id] = trace
                return None
        except (ImportError, Exception):
            pass

        # MLflow
        try:
            import mlflow
            active_run = mlflow.get_active_run()
            if active_run:
                mlflow.log_metrics({"agent_latency_ms": elapsed})
                trace.update({"active_run": active_run, "observability": "mlflow"})
                log.info(f"[TRACER] T={elapsed:.0f}ms  mlflow_captured  run_id={run_id}")
                self._traces[run_id] = trace
                return None
        except (ImportError, Exception):
            pass

        # Fallback: extract from state messages
        trace.update(self._extract_from_state(run_id, elapsed, state.get("messages", [])))
        trace["observability"] = "local"
        log.info(
            f"[TRACER] T={elapsed:.0f}ms  run_complete  run_id={run_id}  "
            f"tools={trace.get('tools_called')}  llm_turns={trace.get('llm_turns')}"
        )
        self._traces[run_id] = trace
        return None

    @staticmethod
    def _extract_from_state(run_id: str, elapsed_ms: float, messages: list) -> dict:
        """Walk message list → structured trace. Zero deps, 0ms overhead."""
        question, answer, tool_calls, tool_results, llm_turns = "", "", [], [], 0
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human" and not question:
                question = str(msg.content)
            elif isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_calls.append({"name": tc.get("name", "unknown"), "args": tc.get("args", {})})
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

    def get_trace(self, run_id: str) -> Optional[dict]:
        """Gateway calls this to attach trace_id to HTTP response."""
        return self._traces.get(run_id)
