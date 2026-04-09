"""
prompt.py — Context Engineering (@dynamic_prompt + Bedrock Prompt Management)
================================================================================
Builds the full system prompt dynamically each LLM call by fetching the base
template from AWS Bedrock Prompt Management and injecting runtime variables.

WHY Bedrock Prompt Management (not hardcoded strings):
  Prompts are content, not code. Clinical writers can edit, version, and A/B
  test templates in the Bedrock console without touching the codebase or
  triggering a deployment. Version IDs allow instant rollback.

Bedrock template uses {{variable}} placeholders:
  {{domain_frame}}     — clinical vs general framing string
  {{max_tool_calls}}   — integer cap from tools/__init__.py

WHY episodic context is NOT injected here:
  EpisodicMemoryMiddleware.before_agent() injects relevant past interactions
  as a SystemMessage directly into state before the LLM call. This means:
    - One Pinecone search per turn (not two)
    - Semantic relevance search (query=current_question) not recency fetch
    - Context is visible in state history and summarized by SummarizationMiddleware
  The old approach of injecting into {{episodic_context}} here required a
  second Pinecone call and used query="" (recency, not relevance).

SSM parameters consumed (via core.aws):
  /{APP_NAME}/{env}/bedrock/prompt_id
  /{APP_NAME}/{env}/bedrock/prompt_version

SSM is read on every call so a version update takes effect without restart.
The template fetch itself is cached by (id, version) — a version bump causes
a fresh Bedrock fetch on the next request.
"""

import logging
from functools import lru_cache
from typing import Any

from langchain.agents.middleware import dynamic_prompt

from core import aws
from agent.tools import MAX_TOOL_CALLS_PER_REQUEST

log = logging.getLogger(__name__)

_APP_NAME = "clinical-trial-agent"


@lru_cache(maxsize=2)
def _fetch_prompt_template(prompt_id: str, prompt_version: str) -> str:
    """
    Fetch and cache the Bedrock prompt template by (id, version).

    lru_cache(maxsize=2) keeps the last two (id, version) pairs warm.
    A version bump in SSM causes a cache miss and a fresh Bedrock fetch.
    """
    log.info(f"[PROMPT] Fetching Bedrock template  id={prompt_id}  version={prompt_version}")
    return aws.get_bedrock_prompt(prompt_id, prompt_version)


def _get_template() -> str:
    """
    Read prompt_id + prompt_version from SSM and return the cached template.

    SSM is read on every call (not cached) so version updates take effect
    immediately without a process restart.

    Local mode: skips SSM entirely — aws.get_bedrock_prompt() returns
    _LOCAL_SYSTEM_PROMPT which includes the MANDATORY CLARIFICATION RULE.
    """
    if aws.is_local():
        # "local" / "0" are sentinels — get_bedrock_prompt returns _LOCAL_SYSTEM_PROMPT
        return _fetch_prompt_template("local", "0")

    env            = aws.get_env()
    prompt_id      = aws.get_ssm_parameter(f"/{_APP_NAME}/{env}/bedrock/prompt_id",     with_decryption=False)
    prompt_version = aws.get_ssm_parameter(f"/{_APP_NAME}/{env}/bedrock/prompt_version", with_decryption=False)
    return _fetch_prompt_template(prompt_id, prompt_version)


def _build_domain_frame(domain: str) -> str:
    """
    Return the domain-specific framing string injected into {{domain_frame}}.

    Pharma framing aligns with OutputGuardrailMiddleware constraints:
      "evidence-based"               -> Layer 2 faithfulness judge
      "no treatment recommendations" -> Layer 1 regex patterns
      "faithfulness non-negotiable"  -> primes LLM to stay close to sources
    """
    if domain == "pharma":
        return (
            "You are operating in a PHARMA / CLINICAL TRIAL domain. "
            "All answers must be evidence-based, cite retrieved sources, and include "
            "appropriate clinical disclaimers. Never provide direct treatment "
            "recommendations. Faithfulness to retrieved context is non-negotiable."
        )
    return (
        "You are a knowledgeable research assistant. "
        "Always retrieve evidence before answering. Cite sources. Be precise."
    )


@dynamic_prompt
def context_aware_prompt(request: Any) -> str:
    """
    Dynamically built system prompt — re-evaluated before every LLM call.

    Flow:
      1. Extract domain from runtime.context.
      2. Read prompt_id + prompt_version from SSM.
      3. Fetch (or return cached) template from Bedrock.
      4. Build domain_frame string.
      5. Substitute {{variables}} and return final prompt string.

    Episodic context is NOT injected here — EpisodicMemoryMiddleware.before_agent()
    handles it by injecting a SystemMessage into state directly. This avoids a
    second Pinecone call and ensures relevance-based retrieval (not recency).

    Fallback: if Bedrock is unreachable a minimal inline prompt is used so the
    agent continues to function. MUST include the MANDATORY CLARIFICATION RULE —
    without it HITL is silently broken (LLM answers in plain text).
    """
    ctx    = getattr(getattr(request, "runtime", None), "context", {}) or {}
    domain = ctx.get("domain", "general")

    try:
        template = _get_template()
    except Exception as exc:
        log.error(f"[PROMPT] Bedrock fetch failed ({exc}) — using emergency fallback prompt")
        # Keep in sync with _LOCAL_SYSTEM_PROMPT in core/aws.py
        template = (
            "{{domain_frame}}\n\n"
            "You are an expert clinical research assistant with deep knowledge of "
            "pharmaceutical drug development, clinical trial design, regulatory "
            "frameworks (FDA, EMA, ICH), and evidence-based medicine.\n\n"
            "CORE BEHAVIOUR:\n"
            "- Always retrieve evidence before answering. Never answer from memory alone.\n"
            "- Cite the specific source, trial name, or document for every clinical claim.\n"
            "- Be precise with numbers — dosages, p-values, endpoints, sample sizes matter.\n\n"
            "CLARIFICATION RULE — MANDATORY:\n"
            "When the request is ambiguous, you MUST call the ask_user_input tool.\n"
            "Do NOT ask clarifying questions in plain text — ALWAYS use the tool.\n"
            "Failing to call the tool means the user cannot respond interactively.\n"
            "Use ask_user_input when: the trial name or drug is ambiguous, the question\n"
            "could refer to multiple phases or indications, or the user intent is unclear.\n\n"
            "TOOL USAGE: Maximum {{max_tool_calls}} tool calls per request."
        )

    domain_frame = _build_domain_frame(domain)

    return (
        template
        .replace("{{domain_frame}}",   domain_frame)
        .replace("{{max_tool_calls}}", str(MAX_TOOL_CALLS_PER_REQUEST))
    )