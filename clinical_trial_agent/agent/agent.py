"""
agent.py — Agent Assembly
==========================
Wires all pillars together into a compiled LangChain agent.

Pillar mapping:
  01 Tools         → agent/tools/            (search, graph, summariser, chart, hitl)
  02 Middleware    → agent/middleware/        (9 layers, see middleware/__init__.py)
  03 Context Eng.  → agent/prompt.py         (@dynamic_prompt + Bedrock Prompt Management)
  04 Schema        → agent/schema.py         (AgentContext TypedDict)
  05 Checkpointer  → PostgresSaver           (durable HITL pause/resume across restarts)
  06 Store         → PineconeStore           (episodic memory + @dynamic_prompt retrieval)
  07 Cache         → SemanticCache(Pinecone) (persistent semantic cache in middleware)

Gateway responsibilities (NOT here):
  ✓ Auth / JWT validation      → vs_platform/gateway/auth.py
  ✓ Rate limiting              → vs_platform/gateway/rate_limiter.py
  ✓ Prompt injection check     → vs_platform/gateway/injection.py

Credential sourcing (NOT here):
  ✓ Pinecone API key + index   → core.aws.init_pinecone_index()   via SSM
  ✓ Postgres DSN               → core.aws.init_postgres_url()     via Secrets Manager
  ✓ Bedrock prompt template    → core.aws.get_bedrock_prompt_from_ssm() via SSM + Bedrock

Prerequisites (local or AWS):
  - AWS credentials configured (aws configure / IAM role)
  - SSM parameters set (see README.md for full list)
  - Pinecone index created (dimension=1536, metric=cosine)
  - Postgres running with a database created
  - Bedrock prompt created in AWS console

WHY build_agent() is a factory function (not a module-level singleton):
  Each call produces independent store, cache, and checkpointer instances.
  Test isolation: every test fixture gets a fresh agent with empty state.
  Multi-domain: pharma and general agents share no state.

WHY two separate LLM instances:
  create_agent uses "gpt-4o" for main reasoning.
  OutputGuardrailMiddleware uses "gpt-4o-mini" at temperature=0 —
  cheaper deterministic judge for faithfulness scoring.

WHY store is passed to both build_stack() and create_agent():
  EpisodicMemoryMiddleware writes Q&A pairs to the store.
  create_agent(store=...) makes the same instance available to
  @dynamic_prompt via request.runtime.store. Both must be the same object.

WHY both store and cache share the same Pinecone index:
  Pinecone is billed per index, not per namespace. One index with two
  namespace groups (episodic__ and cache_) minimises cost while keeping
  the two datasets logically isolated.
"""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver

from core import aws
from core.cache import SemanticCache
from core.pinecone_store import PineconeStore
from agent.middleware import build_stack
from agent.schema import AgentContext
from agent.tools import ALL_TOOLS

log = logging.getLogger(__name__)


def build_agent(domain: str = "general") -> Any:
    """
    Assemble and return a compiled LangChain agent with all pillars wired together.

    Fetches all credentials at call time via core.aws — no credentials live here.

    Args:
        domain: "pharma" | "general"
                Controls SemanticCache threshold and system prompt framing.

    Returns:
        Compiled LangChain agent ready for invoke() in run.py.

    Raises:
        botocore.exceptions.ClientError  — SSM or Secrets Manager access denied.
        EnvironmentError                 — required SSM parameter missing.
        pinecone.exceptions.PineconeException — Pinecone index not found.
        psycopg.OperationalError         — Postgres unreachable or wrong credentials.
    """
    # Shared embedder — one instance reuses the HTTP connection pool across
    # both PineconeStore and SemanticCache.
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

    # PostgresSaver — durable HITL checkpointer.
    # from_conn_string() returns a context manager in langgraph-checkpoint-postgres>=3.x.
    # Use the connection string directly with psycopg instead.
    # setup() creates checkpoint tables on first run (idempotent thereafter).
    import psycopg
    conn_string  = aws.init_postgres_url()
    conn         = psycopg.connect(conn_string, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    safety_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    middleware_stack = build_stack(
        domain=domain,
        store=store,
        safety_llm=safety_llm,
        cache=cache,
    )

    agent = create_agent(
        model="gpt-4o",
        tools=ALL_TOOLS,
        middleware=middleware_stack,
        store=store,
        checkpointer=checkpointer,
        context_schema=AgentContext,
        # No system_prompt= here — context_aware_prompt (@dynamic_prompt)
        # builds the prompt dynamically each turn from Bedrock Prompt Management.
        # HITL is handled by HumanInTheLoopMiddleware(interrupt_on={ask_user_input: True})
        # in the middleware stack — not by interrupt_before at the graph level.
    )

    log.info(
        f"[AGENT] Built  domain={domain}"
        f"  tools={[t.name for t in ALL_TOOLS]}"
        f"  middleware={len(middleware_stack)}"
        f"  store=pinecone  cache=pinecone(cache_{domain})"
        f"  checkpointer=postgres"
    )
    return agent