"""
agent/middleware/__init__.py — Clinical Trial Agent Middleware Stack
=====================================================================
Assembles the full 9-layer stack from:
  core/middleware/   — domain-agnostic layers (tracer, cache, episodic)
  agent/middleware/  — pharma-domain layers (PII, content filter, guardrails)

Execution order:
  before_agent: TOP → BOTTOM  (ingress)
  after_agent:  BOTTOM → TOP  (egress — output guard closest to user)

Stack:
  1. TracerMiddleware          core    — cross-cutting observability
  2. DomainPIIMiddleware       domain  — HIPAA-aligned PII scrubbing
  3. ContentFilterMiddleware   domain  — pharma toxic content check
  4. SemanticCacheMiddleware   core    — HIT skips everything below
  5. EpisodicMemoryMiddleware  core    — enrich context, store after
  6. SummarizationMiddleware   core    — compress long history
  7. HumanInTheLoopMiddleware  core    — single HITL gate
  8. ActionGuardrailMiddleware domain  — tool call limits
  9. OutputGuardrailMiddleware domain  — code-first → faithfulness → contradiction
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware

# Domain-agnostic middleware from vs-agent-core
from core.aws import get_trace_table_name
from core.middleware.tracer import TracerMiddleware
from core.middleware.semantic_cache import SemanticCacheMiddleware
from core.middleware.episodic_memory import EpisodicMemoryMiddleware
from core.cache import SemanticCache

# Pharma-domain middleware — lives here
from agent.middleware.pii import DomainPIIMiddleware
from agent.middleware.content_filter import ContentFilterMiddleware
from agent.middleware.action_guardrail import ActionGuardrailMiddleware
from agent.middleware.output_guardrail import OutputGuardrailMiddleware

def build_stack(domain: str, store, safety_llm, cache: SemanticCache) -> list:
    """
    Assemble and return the ordered 9-layer middleware stack.

    Args:
        domain:     "pharma" | "general"
        store:      PineconeStore instance — shared with create_agent(store=...)
        safety_llm: LLM for OutputGuardrailMiddleware faithfulness judging
        cache:      SemanticCache (Pinecone-backed) — injected from build_agent()

    Returns:
        Ordered list of middleware instances ready for create_agent(middleware=...).
    """
    return [
        TracerMiddleware(dynamodb_table_name=get_trace_table_name()),
        DomainPIIMiddleware(),
        ContentFilterMiddleware(),
        SemanticCacheMiddleware(cache=cache),
        EpisodicMemoryMiddleware(store=store),
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 3_000),
            keep=("messages", 10),
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={"ask_user_input": True},
        ),
        #ActionGuardrailMiddleware(),
        OutputGuardrailMiddleware(
            llm=safety_llm,
            faithfulness_threshold=0.85,
            confidence_threshold=0.75,
        ),
    ]


__all__ = [
    "TracerMiddleware",
    "DomainPIIMiddleware",
    "ContentFilterMiddleware",
    "SemanticCacheMiddleware",
    "EpisodicMemoryMiddleware",
    "ActionGuardrailMiddleware",
    "OutputGuardrailMiddleware",
    "build_stack",
]