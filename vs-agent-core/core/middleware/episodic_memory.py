"""
episodic_memory.py — EpisodicMemoryMiddleware
===============================================
Uses any LangGraph BaseStore-compatible store for episodic storage.
In production this is PineconeStore; in tests InMemoryStore is used.

Episodic ENRICHES context — HIT adds past interactions to the prompt,
but tools and LLM still run (unlike cache which skips everything).

Scope:  per user, persisted across sessions via PineconeStore in production.
Write:  fire-and-forget daemon thread after response is sent.


Storage decision:
  The LLM itself tags every response with EPISODIC: YES | NO.
  WHY LLM decides (not keywords):
    "What dose for eGFR 25?" is user-specific → YES
    "What is the normal eGFR range?" is generic → NO
    A keyword list cannot make that distinction reliably.
"""

import hashlib
import logging
import re
import threading
import time
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

log = logging.getLogger(__name__)


class EpisodicMemoryMiddleware(BaseAgentMiddleware):

    _EPISODIC_TAG = re.compile(r"EPISODIC:\s*(YES|NO)", re.IGNORECASE)

    def __init__(self, store: BaseStore):
        super().__init__()
        self._store = store  # Shared with create_agent(store=...)

    def _get_user_id(self, runtime: Runtime) -> str:
        ctx = getattr(runtime, "context", {}) or {}
        return ctx.get("user_id", "anonymous")

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log episodic lookup — results flow into @dynamic_prompt via the Store."""
        user_id = self._get_user_id(runtime)
        try:
            items = self._store.search(("episodic", user_id), query="", limit=3)
            log.info(f"[EPISODIC] Searched  user={user_id}  hits={len(items)}")
        except Exception:
            log.info("[EPISODIC] Search unavailable — proceeding without episodic context")
        return None

    @staticmethod
    def _parse_storage_decision(answer: str) -> tuple[bool, str]:
        """
        Read the EPISODIC: YES/NO tag the LLM included in its response.
        Returns (should_store, clean_answer_without_tag).
        """
        match = EpisodicMemoryMiddleware._EPISODIC_TAG.search(answer)
        if not match:
            return False, answer  # No tag — conservative default: don't store

        should_store = match.group(1).upper() == "YES"
        # Strip the tag line — it's internal metadata, user should never see it
        clean = re.sub(r"\nEPISODIC:.*$", "", answer, flags=re.IGNORECASE | re.MULTILINE).strip()
        return should_store, clean

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        user_id  = self._get_user_id(runtime)
        messages = state.get("messages", [])

        question, answer = "", ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not answer:
                answer = str(msg.content)
            elif hasattr(msg, "type") and msg.type == "human" and not question:
                question = str(msg.content)
            if question and answer:
                break

        if not (question and answer):
            return None

        should_store, clean_answer = self._parse_storage_decision(answer)

        # Strip the EPISODIC tag from the message the user sees
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                msg.content = clean_answer
                break

        if not should_store:
            log.info(f"[EPISODIC] LLM tagged NO — skipping store  user={user_id}")
            return None

        log.info(f"[EPISODIC] LLM tagged YES — storing  user={user_id}")
        threading.Thread(
            target=self._store_sync,
            args=(user_id, question, clean_answer),
            daemon=True,
        ).start()
        return None

    def _store_sync(self, user_id: str, question: str, answer: str):
        try:
            entry_id = hashlib.md5(f"{user_id}{question}".encode()).hexdigest()[:12]
            self._store.put(
                ("episodic", user_id),
                entry_id,
                {"text": f"Q: {question}\nA: {answer[:300]}", "ts": time.time()},
            )
            log.info(f"[EPISODIC] Stored  user={user_id}  id={entry_id}")
        except Exception as exc:
            log.warning(f"[EPISODIC] Store failed: {exc}")
