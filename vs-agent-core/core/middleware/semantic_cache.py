"""
semantic_cache.py — SemanticCacheMiddleware
=============================================
Semantic cache check (before) and store (after).
Cache REPLACES work — HIT means NO episodic search, NO tools, NO LLM call.

The SemanticCache (Pinecone-backed) instance is injected at construction from
build_agent() — this middleware owns the lookup/store lifecycle only, not the
cache implementation or embedding logic.
"""

import logging
import threading
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.cache import SemanticCache

log = logging.getLogger(__name__)


class SemanticCacheMiddleware(BaseAgentMiddleware):
    """
    Two-hook middleware that short-circuits the entire agent on a cache hit.

    The SemanticCache instance is injected at construction (not created here)
    so build_agent() controls the Pinecone index, namespace, embedder, and
    threshold in one place. This middleware owns only the lookup → short-circuit
    and write lifecycle, not the cache implementation.

    Instance state:
      _cache         — Pinecone-backed SemanticCache (injected, not created here).
      _last_question — run_id → question string bridge between before_agent
                       (where the question is known) and after_agent (where the
                       answer is known). Entries are pop()'d in after_agent to
                       prevent unbounded growth.

    The embedding step moved from this middleware into SemanticCache.lookup() and
    SemanticCache.store(). This middleware passes raw question strings — the cache
    owns the embed-then-search / embed-then-upsert pipeline internally.
    """

    def __init__(self, cache: SemanticCache):
        super().__init__()
        self._cache          = cache
        self._last_question: dict[str, str] = {}

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        CACHE LOOKUP — pass the user question to the cache and short-circuit on HIT.

        Flow:
          1. Guard-clause exits (no messages, not a HumanMessage, empty string).
          2. Store question in _last_question[run_id] for after_agent to use.
          3. Call cache.lookup(question) — embeds internally, queries Pinecone.
          4a. HIT  → return cached AIMessage + jump_to="end" (zero LLM tokens).
          4b. MISS → return None, execution continues to the next middleware.

        WHY the question is stored before the lookup (step 2 before step 3):
          If lookup() raises, after_agent still finds the question in _last_question
          and stores the fresh LLM answer — turning a lookup error into a cache
          population opportunity rather than silent data loss.

        WHY jump_to="end" on HIT:
          Skips EpisodicMemoryMiddleware, SummarizationMiddleware,
          HumanInTheLoopMiddleware, ActionGuardrailMiddleware,
          OutputGuardrailMiddleware. Cached answers were guardrail-checked when
          first generated — re-evaluating them adds ~1.6s latency for no benefit.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not (hasattr(last_msg, "type") and last_msg.type == "human"):
            return None

        user_content = str(last_msg.content).strip()
        if not user_content:
            return None
        user_id = runtime.context.user_id
        run_id = self._get_run_id(runtime)
        self._last_question[run_id] = user_content

        try:
            cached = self._cache.lookup(user_content,user_id)
            if cached:
                log.info("[CACHE_MIDDLEWARE] HIT — returning cached answer, skipping all work")
                return {"messages": [AIMessage(content=cached)], "jump_to": "end"}
        except Exception as exc:
            log.warning(f"[CACHE_MIDDLEWARE] Lookup failed ({exc}) — proceeding without cache")

        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        CACHE WRITE — store the LLM's answer against the question from before_agent.

        Spawns a daemon thread so the response reaches the caller without waiting
        for the Pinecone upsert (~50ms embedding + ~100ms network).

        pop() on _last_question cleans up the bridge entry in one operation,
        preventing unbounded growth in long-running processes.
        """
        user_id = runtime.context.user_id
        run_id   = self._get_run_id(runtime)
        question = self._last_question.pop(run_id, "")
        if not question:
            return None

        answer = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if question and answer:
            threading.Thread(
                target=self._store_sync,
                args=(question, answer, user_id),
                daemon=True,
            ).start()

        return None

    def _store_sync(self, question: str, answer: str, user_id:str, ttl: int = 3_600) -> None:
        """
        Background write — called from daemon thread in after_agent.

        Passes the raw question string to cache.store() which handles embedding
        and Pinecone upsert internally. ttl=3_600 (1 hour) balances freshness
        against hit rate for clinical data that may be updated with new trial
        results or guideline revisions.
        """
        try:
            self._cache.store(question, answer, user_id,ttl=ttl)
        except Exception as exc:
            log.warning(f"[CACHE_MIDDLEWARE] Store failed: {exc}")
