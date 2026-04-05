"""
agent.py — Agent Assembly
==========================
Wires all pillars together into a compiled LangChain agent.

Pillar mapping:
  01 Tools         → agent/tools/            (search, graph, summariser, chart, hitl)
  02 Middleware    → agent/middleware/        (9 layers, see middleware/__init__.py)
  03 Context Eng.  → agent/prompt.py         (@dynamic_prompt)
  04 Schema        → agent/schema.py         (AgentContext TypedDict)
  05 Checkpointer  → PostgresSaver           (durable HITL pause/resume across restarts)
  06 Store         → PineconeStore           (episodic memory + @dynamic_prompt retrieval)
  07 Cache         → SemanticCache(Pinecone) (persistent semantic cache in middleware)

Gateway responsibilities (NOT here — live in gateway.py):
  ✓ Auth / JWT validation      → gateway._auth()
  ✓ Rate limiting              → gateway._rate_ok()
  ✓ Prompt injection check     → gateway.run_input_guardrails()

Credential sourcing (NOT here — live in agent/aws.py):
  ✓ Pinecone API key + index   → aws.init_pinecone_index()  via SSM Parameter Store
  ✓ Postgres DSN               → aws.init_postgres_url()    via Secrets Manager

WHY build_agent() is a factory function (not a module-level singleton):
  Each call produces independent store, cache, and checkpointer instances.
  Test isolation: every test fixture gets a fresh agent with empty state.
  Multi-domain: pharma and general agents run in the same process without
  sharing checkpoints, episodic memory, or cache state.

WHY two separate LLM instances:
  create_agent uses "gpt-4o" for main reasoning — highest capability for
  clinical Q&A and tool selection.
  OutputGuardrailMiddleware uses "gpt-4o-mini" at temperature=0 for
  faithfulness/contradiction scoring — cheaper deterministic judge.

WHY store is passed to both build_stack() and create_agent():
  EpisodicMemoryMiddleware (in build_stack) writes Q&A pairs to the store.
  create_agent(store=...) makes the same store instance available to
  @dynamic_prompt via request.runtime.store. Both must reference the same
  object — different instances would produce a write/read split.

WHY both store and cache share the same Pinecone index:
  Pinecone is billed per index, not per namespace. One index with two
  namespace groups (episodic__ prefix and cache_ prefix) minimises cost
  while keeping the two datasets logically isolated.

pip install:
  langgraph-checkpoint-postgres psycopg[binary,pool]
  pinecone-client boto3 langchain-openai
"""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver

from core import aws
from core.cache import SemanticCache
from agent.middleware import build_stack
from core.pinecone_store import PineconeStore
from agent.schema import AgentContext
from agent.tools import ALL_TOOLS

log = logging.getLogger(__name__)


def build_agent(domain: str = "general") -> Any:
    """
    Assemble and return a compiled LangChain agent with all pillars wired together.

    Fetches AWS credentials at call time via agent.aws — no credentials live here.

    Args:
        domain: Controls two domain-aware components:
                  "pharma"  → SemanticCache threshold=0.97, namespace="cache_pharma",
                              clinical system prompt framing in context_aware_prompt
                  "general" → SemanticCache threshold=0.88, namespace="cache_general",
                              research assistant framing

    Returns:
        Compiled LangChain agent ready for invoke_with_hitl() in run.py.

    Raises:
        botocore.exceptions.ClientError — SSM or Secrets Manager access denied.
        pinecone.exceptions.PineconeException — Pinecone index not found.
        psycopg.OperationalError — Postgres unreachable or credentials invalid.
    """
    # Shared embedder — one instance reuses the HTTP connection pool across
    # both PineconeStore (episodic writes/reads) and SemanticCache (lookup/store).
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # Single Pinecone index for both store and cache, isolated by namespace.
    pinecone_index = aws.init_pinecone_index()

    # PineconeStore — LangGraph BaseStore adapter.
    # Passed to both build_stack() and create_agent() — must be the same instance.
    store = PineconeStore(index=pinecone_index, embedder=embedder)

    # SemanticCache — domain-namespaced, TTL-filtered.
    cache = SemanticCache(
        index=pinecone_index,
        embedder=embedder,
        similarity_threshold=0.97 if domain == "pharma" else 0.88,
        namespace=f"cache_{domain}",
    )

    # PostgresSaver — durable HITL checkpointer.
    # setup() creates checkpoint tables on first run (idempotent on subsequent runs).
    checkpointer = PostgresSaver.from_conn_string(aws.init_postgres_url())
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
        # NOTE: no system_prompt= — context_aware_prompt (@dynamic_prompt) builds
        # the prompt dynamically each turn, reading episodic context from runtime.store.
    )

    log.info(
        f"[AGENT] Built  domain={domain}"
        f"  tools={[t.name for t in ALL_TOOLS]}"
        f"  middleware={len(middleware_stack)}"
        f"  store=pinecone  cache=pinecone(cache_{domain})"
        f"  checkpointer=postgres"
    )
    return agent
