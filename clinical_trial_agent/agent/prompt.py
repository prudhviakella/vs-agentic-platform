"""
prompt.py — System Prompt Builder
===================================
Fetches the system prompt from AWS Bedrock Prompt Management and substitutes
runtime variables.

WHY Bedrock Prompt Management?
  Prompts are content, not code. Clinical writers can edit, version, and
  A/B test prompt templates in the Bedrock console without touching the
  codebase or triggering a deployment. Version IDs allow instant rollback.

Bedrock template uses {{variable}} placeholders:
  {{domain_frame}}   — pharma vs general framing (built at agent creation time)
  {{max_tool_calls}} — integer cap from tools/__init__.py

Episodic context is NOT a placeholder here — EpisodicMemoryMiddleware injects
a SystemMessage directly into state before each model call.
"""

import logging
from core import aws
from agent.tools import MAX_TOOL_CALLS_PER_REQUEST

log = logging.getLogger(__name__)

_APP_NAME = "clinical-trial-agent"


def build_system_prompt(domain: str) -> str:
    """
    Fetch the Bedrock prompt template and substitute domain_frame and
    max_tool_calls. Called once at agent creation time — domain is fixed
    per agent instance.
    """
    if domain == "pharma":
        domain_frame = (
            "You are operating in a PHARMA / CLINICAL TRIAL domain. "
            "All answers must be evidence-based, cite retrieved sources, and include "
            "appropriate clinical disclaimers. Never provide direct treatment "
            "recommendations. Faithfulness to retrieved context is non-negotiable."
        )
    else:
        domain_frame = (
            "You are a knowledgeable research assistant. "
            "Always retrieve evidence before answering. Cite sources. Be precise."
        )

    template = aws.get_bedrock_prompt(_APP_NAME)

    return (
        template
        .replace("{{domain_frame}}",   domain_frame)
        .replace("{{max_tool_calls}}", str(MAX_TOOL_CALLS_PER_REQUEST))
    )