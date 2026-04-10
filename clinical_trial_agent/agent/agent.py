"""
agent.py — Agent Assembly
==========================
Wires all pillars together into a compiled LangChain agent.

Pillars:
  01 Tools         → agent/tools/
  02 Middleware    → agent/middleware/        (9 layers)
  03 System Prompt → agent/prompt.py         (from Bedrock Prompt Management)
  04 Schema        → agent/schema.py
  05 Checkpointer  → PostgresSaver
  06 Store         → PineconeStore
  07 Cache         → SemanticCache (Pinecone)
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
from agent.prompt import build_system_prompt
from agent.schema import AgentContext
from agent.tools import ALL_TOOLS

log = logging.getLogger(__name__)


def build_agent(domain: str = "general", use_postgres: bool = False) -> Any:
    embedder       = OpenAIEmbeddings(model="text-embedding-3-small")
    pinecone_index = aws.init_pinecone_index()
    store          = PineconeStore(index=pinecone_index, embedder=embedder)

    cache = SemanticCache(
        index=pinecone_index,
        embedder=embedder,
        similarity_threshold=0.97 if domain == "pharma" else 0.88,
        namespace=f"cache_{domain}",
    )

    if use_postgres:
        import psycopg
        conn         = psycopg.connect(aws.init_postgres_url(), autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        checkpointer_label = "postgres"
    else:
        checkpointer       = MemorySaver()
        checkpointer_label = "memory"

    safety_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_agent(
        model="gpt-4o",
        tools=ALL_TOOLS,
        system_prompt=build_system_prompt(domain),
        middleware=build_stack(
            domain=domain,
            store=store,
            safety_llm=safety_llm,
            cache=cache,
        ),
        store=store,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )

    log.info(
        f"[AGENT] Built  domain={domain}"
        f"  tools={[t.name for t in ALL_TOOLS]}"
        f"  checkpointer={checkpointer_label}"
    )
    return agent