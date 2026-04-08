"""
episodic_memory.py — EpisodicMemoryMiddleware
===============================================
Gives the agent memory across sessions — it remembers what a specific user
asked in the past and uses that to enrich the current response.

WHY do we need this?
  Without episodic memory every conversation starts blank.

  Session 1:  User asks "What is the metformin dose for my patient with eGFR 25?"
              Agent gives a detailed answer.

  Session 2:  User asks "Is that dose still safe?"
              Agent: "Which dose? Which patient?" — it has forgotten everything.

  With episodic memory:
  Session 2:  Agent injects prior context into the prompt automatically →
              "Based on our previous discussion about eGFR 25 dosing..."

HOW is this different from SemanticCache?
  SemanticCache  — shared across ALL users, skips the agent entirely on HIT.
                   Stores generic answers ("metformin Phase 3 results").
  EpisodicMemory — PRIVATE per user, agent still runs (just with richer context).
                   Stores user-specific interactions ("this user's patient has eGFR 25").

  Cache  = saves cost.
  Episodic = enables personalisation.

THE CORE IDEA — relevant retrieval, not full state passing:
  The naive approach is to pass the entire conversation history to the LLM.
  This breaks down fast:

    Session 1:   10 messages  →   ~2,000 tokens
    Session 10: 100 messages  →  ~20,000 tokens  ← expensive, hits context limit

  Episodic memory solves this by storing past Q&A pairs in Pinecone and
  retrieving only the TOP 3 MOST RELEVANT ones for the current question:

    User asks current question
          ↓
    Embed question → search Pinecone (episodic__user_abc namespace)
          ↓
    Returns top 3 semantically similar past Q&As
          ↓
    @dynamic_prompt injects them into the system prompt
          ↓
    LLM sees: current question + 3 relevant memories  (~500 tokens, always bounded)

  Two benefits:
    1. Token cost is BOUNDED — 3 sessions or 300 sessions, same token cost.
    2. Relevance over recency — a memory from 20 sessions ago surfaces if it
       is semantically related to today's question. Pure state passing would
       miss it entirely once the context window fills up.

  This is why it is called EPISODIC — borrowed from human psychology.
  Episodic memory in humans is how you recall a specific past experience
  when something in the present triggers it, not by replaying your entire
  life history.

WHY does the LLM decide what to store (not a keyword rule)?
  Consider these two questions — both contain the word "eGFR":
    "What is the normal eGFR range?"          → generic, anyone would ask this → NO
    "What dose for my patient with eGFR 25?"  → specific to this user's patient → YES

  A keyword rule cannot tell them apart. Only the LLM understands context.
  The LLM tags every response with EPISODIC: YES or NO at zero extra cost
  since it is already running anyway.
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
        # Shared with build_agent(store=...) and @dynamic_prompt —
        # writes here are immediately visible to the prompt on the next turn
        self._store = store

    def _get_user_id(self, runtime: Runtime) -> str:
        ctx = getattr(runtime, "context", {}) or {}
        return ctx.get("user_id", "anonymous")

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Log how many episodic hits were found for this user.
        The actual retrieval and prompt injection happens inside @dynamic_prompt
        via the shared store — we just observe here.
        """
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
        Read the EPISODIC: YES/NO tag that the LLM appended to its response.

        The system prompt instructs the LLM to end every answer with this tag.
        We strip the tag before returning the answer to the user — it is
        internal metadata they should never see.

        Returns (should_store, clean_answer_without_tag).
        """
        match = EpisodicMemoryMiddleware._EPISODIC_TAG.search(answer)
        if not match:
            # No tag found — conservative default: don't store
            # (could be a cached answer or a guardrail-blocked response)
            return False, answer

        should_store = match.group(1).upper() == "YES"

        # Remove the EPISODIC tag line so it never reaches the user
        clean = re.sub(
            r"\nEPISODIC:.*$", "", answer,
            flags=re.IGNORECASE | re.MULTILINE
        ).strip()
        return should_store, clean

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        user_id  = self._get_user_id(runtime)
        messages = state.get("messages", [])

        # Walk backwards through messages to find the latest Q&A pair
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

        # Always strip the EPISODIC tag from the message the user sees,
        # regardless of whether we store — it is always internal metadata
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                msg.content = clean_answer
                break

        if not should_store:
            log.info(f"[EPISODIC] LLM tagged NO — skipping store  user={user_id}")
            return None

        log.info(f"[EPISODIC] LLM tagged YES — storing  user={user_id}")

        # Fire-and-forget — response is already on its way to the user,
        # storing to Pinecone happens in the background
        threading.Thread(
            target=self._store_sync,
            args=(user_id, question, clean_answer),
            daemon=True,
        ).start()
        return None

    def _store_sync(self, user_id: str, question: str, answer: str):
        """
        Write the Q&A pair to Pinecone via PineconeStore.put().

        entry_id is an MD5 of user+question — deterministic so asking the
        same question twice overwrites the old entry rather than duplicating it.

        answer is truncated to 300 chars — enough context for future enrichment,
        avoids ballooning Pinecone metadata size.
        """
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