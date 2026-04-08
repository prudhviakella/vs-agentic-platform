"""
semantic_cache.py — SemanticCacheMiddleware
=============================================
Semantic cache check (before) and store (after).
Cache REPLACES work — HIT means NO episodic search, NO tools, NO LLM call.

Bridge design:
  before_agent stores the human question in self._human_message.
  after_agent reads it directly — no run_id, no dict lookup needed.

  Why a scalar is safe here:
  Deployed on AWS AgentCore — each invocation launches a fresh container/
  process with its own SemanticCacheMiddleware instance. There is no shared
  state between requests so self._human_message can never be overwritten by
  a concurrent request. A dict keyed by thread_id is only needed for
  long-running servers (FastAPI/Gunicorn) where one process handles many
  requests with a cached agent instance.

  Fire-and-forget write:
  The Pinecone upsert in after_agent runs in a daemon thread so the response
  is returned to the user immediately. AgentCore warm containers stay alive
  briefly after the handler returns — enough for a fast Pinecone write to
  complete. Using daemon=True ensures the thread does not block container
  shutdown if the write is still in-flight.
"""

import logging
import threading
from typing import Any, Optional

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.cache import SemanticCache

log = logging.getLogger(__name__)


class SemanticCacheMiddleware(BaseAgentMiddleware):
    """
    Two-hook middleware that short-circuits the entire agent on a cache hit.

    Instance state:
      _cache          — injected SemanticCache (Pinecone-backed)
      _human_message  — current question, set in before_agent, read in after_agent
      _user_id        — current user, set in before_agent, read in after_agent
    """

    def __init__(self, cache: SemanticCache):
        super().__init__()
        self._cache:         SemanticCache  = cache
        self._human_message: Optional[str] = None
        self._user_id:       str           = "anonymous"

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        CACHE LOOKUP — check for a cached answer and short-circuit on HIT.

        Stores question and user_id on the instance for after_agent to use.

        HIT  → return cached AIMessage + jump_to="end" (zero LLM tokens).
        MISS → return None, execution continues to the next middleware.
        """
        # ── Extract current human question ─────────────────────────────────
        messages = state.get("messages", [])
        human_messages = [m for m in messages if getattr(m, "type", None) == "human"]

        if not human_messages:
            log.debug("[CACHE_MW] before_agent skip — no human message in state")
            self._human_message = None
            return None

        if len(human_messages) > 1:
            # Multi-turn: answer depends on prior context — skip cache.
            # Caching a context-dependent answer and returning it cold to
            # another user asking the same question would be wrong.
            log.debug(
                f"[CACHE_MW] before_agent skip — multi-turn "
                f"({len(human_messages)} human messages), answer is context-dependent"
            )
            self._human_message = None
            return None

        question = str(human_messages[0].content).strip()
        if not question:
            log.debug("[CACHE_MW] before_agent skip — empty question")
            self._human_message = None
            return None

        # ── Store on instance for after_agent ──────────────────────────────
        user_id = (getattr(runtime, "context", None) or {}).get("user_id", "anonymous")
        self._human_message = question
        self._user_id       = user_id
        log.debug(f"[CACHE_MW] lookup  user={user_id}  question='{question[:80]}'")

        # ── Cache lookup ───────────────────────────────────────────────────
        try:
            cached = self._cache.lookup(question, user_id=user_id)
            if cached:
                log.info(
                    f"[CACHE_MW] HIT  user={user_id}  answer_len={len(cached)}"
                    f" — skipping agent entirely"
                )
                # Clear so after_agent doesn't re-store a cached answer
                self._human_message = None
                return {"messages": [AIMessage(content=cached)], "jump_to": "end"}

            log.debug(f"[CACHE_MW] MISS  user={user_id} — proceeding to agent")

        except Exception as exc:
            log.warning(
                f"[CACHE_MW] lookup error  user={user_id}  error={exc}"
                f" — treating as MISS"
            )

        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        CACHE WRITE — store the LLM answer using the question from before_agent.

        Reads self._human_message set in before_agent.
        Clears it after use so it does not leak into a warm container's next
        invocation (AgentCore may reuse the container for sequential requests).
        """
        question = self._human_message
        user_id  = self._user_id

        # Always clear — prevents leaking into any subsequent warm invocation
        self._human_message = None
        self._user_id       = "anonymous"

        if not question:
            log.debug("[CACHE_MW] after_agent skip — no question stored (multi-turn or cache HIT)")
            return None

        answer = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            log.debug(f"[CACHE_MW] after_agent skip — no AI answer in state  user={user_id}")
            return None

        log.debug(
            f"[CACHE_MW] storing  user={user_id}"
            f"  question='{question[:80]}'  answer_len={len(answer)}"
        )
        threading.Thread(
            target=self._store_sync,
            args=(question, answer, user_id),
            daemon=True,
        ).start()
        return None

    def _store_sync(self, question: str, answer: str, user_id: str, ttl: int = 3_600) -> None:
        """
        Cache write — runs in a daemon thread (fire-and-forget).
        Response is returned to the user before this completes.
        ttl=3_600 (1h): clinical data can change with new trials/guidelines.

        TODO (Phase 2): replace with pub/sub.
          Publish a cache-write event to SNS instead of writing directly.
          A separate Lambda consumer subscribes and calls cache.store().
          Benefits: decoupled, retryable, dead-letter queue for failed writes,
          cache writer scales independently of the agent.

          after_agent becomes:
            self._sns.publish(
                TopicArn=CACHE_WRITE_TOPIC_ARN,
                Message=json.dumps({
                    "question": question,
                    "answer":   answer,
                    "user_id":  user_id,
                    "ttl":      ttl,
                })
            )
        """
        log.debug(f"[CACHE_MW] _store_sync  user={user_id}  ttl={ttl}s")
        try:
            self._cache.store(question, answer, user_id=user_id, ttl=ttl)
            log.info(
                f"[CACHE_MW] store complete  user={user_id}"
                f"  answer_len={len(answer)}  ttl={ttl}s"
            )
        except Exception as exc:
            log.warning(f"[CACHE_MW] store failed  user={user_id}  error={exc}")