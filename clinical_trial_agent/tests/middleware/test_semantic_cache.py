"""
test_semantic_cache.py — SemanticCacheMiddleware Tests
=======================================================
Tests for the cache lookup/short-circuit and write lifecycle.

All Pinecone and embedding calls are mocked — no real API calls.
The mock_cache / mock_cache_hit fixtures are defined in conftest.py.

Constructor change note:
  Old: SemanticCacheMiddleware(domain="pharma")  ← created its own FAISS cache
  New: SemanticCacheMiddleware(cache=mock_cache)  ← cache injected from build_agent()
  These tests verify the new injected-cache interface.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from core.middleware.semantic_cache import SemanticCacheMiddleware


class TestCacheMiss:

    def test_miss_returns_none_and_continues(self, mock_cache, make_state, make_runtime):
        """On MISS, before_agent returns None — execution continues to next middleware."""
        mw    = SemanticCacheMiddleware(cache=mock_cache)
        state = make_state(HumanMessage(content="What is metformin efficacy?"))
        result = mw.before_agent(state, make_runtime())
        assert result is None
        mock_cache.lookup.assert_called_once_with("What is metformin efficacy?")

    def test_miss_stores_question_for_after_agent(self, mock_cache, make_state, make_runtime):
        """Question must be stored in _last_question so after_agent can write it."""
        mw      = SemanticCacheMiddleware(cache=mock_cache)
        runtime = make_runtime()
        state   = make_state(HumanMessage(content="What is metformin efficacy?"))
        mw.before_agent(state, runtime)
        assert runtime.run_id in mw._last_question
        assert mw._last_question[runtime.run_id] == "What is metformin efficacy?"

    def test_empty_messages_returns_none(self, mock_cache, make_state, make_runtime):
        mw     = SemanticCacheMiddleware(cache=mock_cache)
        result = mw.before_agent(make_state(), make_runtime())
        assert result is None
        mock_cache.lookup.assert_not_called()

    def test_non_human_message_returns_none(self, mock_cache, make_state, make_runtime):
        mw     = SemanticCacheMiddleware(cache=mock_cache)
        state  = make_state(AIMessage(content="some AI message"))
        result = mw.before_agent(state, make_runtime())
        assert result is None
        mock_cache.lookup.assert_not_called()

    def test_empty_string_content_returns_none(self, mock_cache, make_state, make_runtime):
        mw     = SemanticCacheMiddleware(cache=mock_cache)
        state  = make_state(HumanMessage(content="   "))
        result = mw.before_agent(state, make_runtime())
        assert result is None
        mock_cache.lookup.assert_not_called()


class TestCacheHit:

    def test_hit_returns_cached_message_and_jumps_to_end(
        self, mock_cache_hit, make_state, make_runtime
    ):
        """On HIT, before_agent returns the cached AIMessage and jump_to='end'."""
        mw     = SemanticCacheMiddleware(cache=mock_cache_hit)
        state  = make_state(HumanMessage(content="What is metformin efficacy?"))
        result = mw.before_agent(state, make_runtime())

        assert result is not None
        assert result.get("jump_to") == "end"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "Cached:" in result["messages"][0].content

    def test_hit_cached_answer_content_matches(self, mock_cache_hit, make_state, make_runtime):
        mw     = SemanticCacheMiddleware(cache=mock_cache_hit)
        state  = make_state(HumanMessage(content="What is metformin efficacy?"))
        result = mw.before_agent(state, make_runtime())
        assert result["messages"][0].content == mock_cache_hit.lookup.return_value


class TestLookupFailure:

    def test_lookup_exception_treated_as_miss(self, make_state, make_runtime):
        """Pinecone errors must not block the request — degrade to MISS."""
        cache = MagicMock()
        cache.lookup.side_effect = RuntimeError("Pinecone timeout")
        mw     = SemanticCacheMiddleware(cache=cache)
        state  = make_state(HumanMessage(content="What is metformin?"))
        result = mw.before_agent(state, make_runtime())
        assert result is None  # degraded to MISS, did not raise


class TestAfterAgentWrite:

    def test_after_agent_calls_cache_store(self, mock_cache, make_state, make_runtime):
        """After a MISS + LLM answer, after_agent must call cache.store()."""
        mw      = SemanticCacheMiddleware(cache=mock_cache)
        runtime = make_runtime()
        state   = make_state(HumanMessage(content="What is metformin?"))

        mw.before_agent(state, runtime)  # populates _last_question

        state_with_answer = make_state(
            HumanMessage(content="What is metformin?"),
            AIMessage(content="Metformin reduces HbA1c by 1.5%."),
        )
        mw.after_agent(state_with_answer, runtime)

        # Give the daemon thread a moment to fire
        import time; time.sleep(0.1)
        mock_cache.store.assert_called_once()
        call_args = mock_cache.store.call_args[0]
        assert call_args[0] == "What is metformin?"
        assert "HbA1c" in call_args[1]

    def test_after_agent_pops_question_from_bridge(self, mock_cache, make_state, make_runtime):
        """_last_question entry must be removed after after_agent runs."""
        mw      = SemanticCacheMiddleware(cache=mock_cache)
        runtime = make_runtime()
        state   = make_state(HumanMessage(content="test question"))

        mw.before_agent(state, runtime)
        assert runtime.run_id in mw._last_question

        state_with_answer = make_state(
            HumanMessage(content="test question"),
            AIMessage(content="test answer"),
        )
        mw.after_agent(state_with_answer, runtime)
        assert runtime.run_id not in mw._last_question

    def test_after_agent_no_question_skips_store(self, mock_cache, make_state, make_runtime):
        """If before_agent was never called (e.g. HIT path), after_agent is a no-op."""
        mw      = SemanticCacheMiddleware(cache=mock_cache)
        runtime = make_runtime()
        state   = make_state(
            HumanMessage(content="question"),
            AIMessage(content="answer"),
        )
        # before_agent never called — _last_question has no entry
        mw.after_agent(state, runtime)
        import time; time.sleep(0.05)
        mock_cache.store.assert_not_called()
