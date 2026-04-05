"""
conftest.py — Shared Pytest Fixtures
======================================
Fixtures shared across all test modules.

Design rules:
  - Fixtures that mock external services (Pinecone, Postgres, OpenAI) use
    unittest.mock — no real API calls in unit/middleware tests.
  - Integration and eval tests may use real LLM calls (mark with @pytest.mark.llm).
  - AgentState and Runtime are always constructed fresh per test — no shared state.
  - SemanticCache and PineconeStore are always mocked in unit/middleware tests.
    Tests that need real Pinecone must be marked @pytest.mark.llm and skipped in CI.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


# ── State / Runtime helpers ────────────────────────────────────────────────────

def make_state(*messages) -> dict:
    """Build a minimal AgentState dict from a list of messages."""
    return {"messages": list(messages)}


def make_runtime(
    user_id: str = "test_user",
    domain: str  = "pharma",
    run_id: str  = "run_test_001",
) -> MagicMock:
    """Build a mock Runtime with context and run_id attributes."""
    runtime         = MagicMock()
    runtime.run_id  = run_id
    runtime.context = {"user_id": user_id, "domain": domain, "session_id": "session_001"}
    runtime.store   = None
    return runtime


# Expose as pytest fixtures
@pytest.fixture
def make_state():   # noqa: F811
    return make_state


@pytest.fixture
def make_runtime():  # noqa: F811
    return make_runtime


# ── Message factories ──────────────────────────────────────────────────────────

@pytest.fixture
def human_msg():
    return HumanMessage(content="What are the efficacy results for metformin?")


@pytest.fixture
def ai_msg():
    return AIMessage(content="Metformin reduces HbA1c by 1.5% on average. [Source: clinical_trials_db]")


@pytest.fixture
def ai_msg_with_medical_action():
    return AIMessage(content="You should take 500mg of metformin twice daily.")


@pytest.fixture
def ai_msg_pii():
    return AIMessage(content="Please contact patient@hospital.com for follow-up.")


@pytest.fixture
def ai_msg_fallback():
    """A message already stamped with the guardrail sentinel."""
    return AIMessage(content="[GUARDRAIL_FALLBACK] I was unable to provide a verified answer.")


@pytest.fixture
def tool_msg_search():
    """Simulated search_tool ToolMessage."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(
        content='{"results": ["Phase 3 RCT: 42% reduction vs placebo"], "source": "vector_db"}',
        tool_call_id="call_001",
        name="search_tool",
    )


@pytest.fixture
def tool_msg_ask_user():
    """Simulated ask_user_input ToolMessage — user answered 'Hyderabad'."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(
        content="Hyderabad",
        tool_call_id="call_002",
        name="ask_user_input",
    )


# ── Runtime fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def runtime():
    return make_runtime()


@pytest.fixture
def runtime_general():
    return make_runtime(domain="general")


# ── Mock LLM fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_high_faith():
    """Mock LLM that returns faithfulness=0.92 (passes threshold)."""
    llm        = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="0.92"))
    return llm


@pytest.fixture
def mock_llm_low_faith():
    """Mock LLM that returns faithfulness=0.30 (fails threshold)."""
    llm        = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="0.30"))
    return llm


@pytest.fixture
def mock_llm_high_consistency():
    """Mock LLM that returns consistency=0.95."""
    llm        = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="0.95"))
    return llm


# ── Store fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def memory_store():
    """
    In-memory BaseStore for unit/middleware tests that exercise EpisodicMemoryMiddleware.

    Uses InMemoryStore (not PineconeStore) so tests run without Pinecone credentials.
    EpisodicMemoryMiddleware accepts any BaseStore — this fixture exploits that.
    Tests that need real Pinecone persistence must be marked @pytest.mark.llm.
    """
    from langgraph.store.memory import InMemoryStore
    return InMemoryStore()


@pytest.fixture
def mock_pinecone_store():
    """
    Mock PineconeStore for tests that need to assert on store.put() / store.search() calls
    without a real Pinecone connection.

    Returns a MagicMock with search() defaulting to [] (no episodic hits).
    Override in individual tests: mock_pinecone_store.search.return_value = [...]
    """
    store = MagicMock()
    store.search.return_value = []
    store.put.return_value    = None
    return store


# ── Cache fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_cache():
    """
    Mock SemanticCache for tests that exercise SemanticCacheMiddleware without
    a real Pinecone index or OpenAI embedding call.

    Default behaviour:
      lookup() → None  (cache MISS — agent proceeds normally)
      store()  → None  (silent success)

    Override for HIT scenario:
      mock_cache.lookup.return_value = "cached answer string"
    """
    cache        = MagicMock()
    cache.lookup.return_value = None   # default: MISS
    cache.store.return_value  = None
    return cache


@pytest.fixture
def mock_cache_hit():
    """
    Mock SemanticCache pre-configured to return a cache HIT.
    Use this to test the jump_to='end' short-circuit path.
    """
    cache        = MagicMock()
    cache.lookup.return_value = "Cached: Metformin reduces HbA1c by 1.5% on average."
    cache.store.return_value  = None
    return cache


# ── pytest markers ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "llm:  mark test as requiring a real LLM API call")
    config.addinivalue_line("markers", "slow: mark test as slow (integration/eval)")
    config.addinivalue_line("markers", "unit: mark test as a fast unit test")
