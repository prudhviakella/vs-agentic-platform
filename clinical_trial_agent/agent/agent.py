"""
agent.py — Agent Assembly
==========================
Wires all pillars together into a compiled LangChain agent.

Pillar mapping:
  01 Tools         -> agent/tools/            (search, graph, summariser, chart, hitl)
  02 Middleware    -> agent/middleware/        (9 layers, see middleware/__init__.py)
  03 Context Eng.  -> agent/prompt.py         (@dynamic_prompt + Bedrock Prompt Management)
  04 Schema        -> agent/schema.py         (AgentContext TypedDict)
  05 Checkpointer  -> PostgresSaver           (durable HITL pause/resume across restarts)
  06 Store         -> PineconeStore           (episodic memory)
  07 Cache         -> SemanticCache(Pinecone) (persistent semantic cache in middleware)

Gateway responsibilities (NOT here):
  Auth, rate limiting, prompt injection  -> vs_platform/gateway/

Credential sourcing (NOT here):
  Pinecone, Postgres, Bedrock            -> core.aws

WHY build_agent() is a factory function (not a module-level singleton):
  Each call produces independent store, cache, and checkpointer instances.
  Test isolation: every test fixture gets a fresh agent with empty state.
  Multi-domain: pharma and general agents share no state.

WHY use_postgres=False is the default:
  PostgresSaver persists checkpoints across restarts — correct for production.
  But run.py uses hardcoded thread_ids. Re-running it would append messages
  on top of old checkpoints causing stale tool_call_id errors.
  MemorySaver is wiped on exit — each run starts clean.

WHY two separate LLM instances:
  create_agent uses gpt-4o for main reasoning.
  OutputGuardrailMiddleware uses gpt-4o-mini at temperature=0 —
  cheaper deterministic judge for faithfulness scoring.

WHY store is passed to both build_stack() and create_agent():
  EpisodicMemoryMiddleware writes Q&A pairs to the store via middleware.
  create_agent(store=...) makes the same instance available to
  @dynamic_prompt via request.runtime.store.
  Both must be the same object so writes are immediately visible.

WHY both store and cache share the same Pinecone index:
  Pinecone is billed per index, not per namespace. One index with two
  namespace groups (episodic__ and cache_) minimises cost while keeping
  the two datasets logically isolated.
"""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from core import aws
from core.cache import SemanticCache
from core.pinecone_store import PineconeStore
from agent.middleware import build_stack
from agent.prompt import context_aware_prompt
from agent.schema import AgentContext
from agent.tools import ALL_TOOLS

log = logging.getLogger(__name__)


def build_agent(domain: str = "general", use_postgres: bool = False) -> Any:
    """
    Assemble and return a compiled LangChain agent with all pillars wired together.

    Args:
        domain:       "pharma" | "general"
                      Controls SemanticCache threshold and system prompt framing.
        use_postgres: True  -> PostgresSaver (durable, survives restarts) — production.
                      False -> MemorySaver   (in-process RAM, wiped on exit) — run.py/tests.

    Returns:
        Compiled LangChain agent ready for invoke().
    """
    # Shared embedder — one instance reuses the HTTP connection pool
    # across both PineconeStore and SemanticCache.
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # Single Pinecone index for both store and cache, isolated by namespace.
    pinecone_index = aws.init_pinecone_index()

    # PineconeStore — LangGraph BaseStore adapter for episodic memory.
    # Passed to both build_stack() and create_agent() — must be the same instance.
    store = PineconeStore(index=pinecone_index, embedder=embedder)

    # SemanticCache — domain-namespaced, TTL-filtered Pinecone-backed cache.
    cache = SemanticCache(
        index=pinecone_index,
        embedder=embedder,
        similarity_threshold=0.97 if domain == "pharma" else 0.88,
        namespace=f"cache_{domain}",
    )

    # Checkpointer:
    #   PostgresSaver — durable, survives restarts, required for production HITL.
    #   MemorySaver   — in-process RAM, clean slate every run, for run.py/tests.
    if use_postgres:
        import psycopg
        conn_string  = aws.init_postgres_url()
        conn         = psycopg.connect(conn_string, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        checkpointer_label = "postgres"
    else:
        checkpointer       = MemorySaver()
        checkpointer_label = "memory"

    safety_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    middleware_stack = build_stack(
        domain=domain,
        store=store,
        safety_llm=safety_llm,
        cache=cache,
    )

    # context_aware_prompt is decorated with @dynamic_prompt which converts it
    # into middleware (uses wrap_model_call internally). It must be in the
    # middleware list — create_agent() has no prompt= parameter in LangChain 1.0.
    # Position 0 — runs first before any model call, sets the system prompt.
    middleware_stack = [context_aware_prompt] + middleware_stack

    agent = create_agent(
        model="gpt-4o",
        tools=ALL_TOOLS,
        middleware=middleware_stack,
        store=store,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )

    log.info(
        f"[AGENT] Built  domain={domain}"
        f"  tools={[t.name for t in ALL_TOOLS]}"
        f"  middleware={len(middleware_stack)}"
        f"  store=pinecone  cache=pinecone(cache_{domain})"
        f"  checkpointer={checkpointer_label}"
    )
    return agent