"""
content_filter.py — ContentFilterMiddleware
=============================================
DOMAIN-SPECIFIC input guardrail — lives in AGENT middleware, not gateway.

WHY here and not gateway:
  Gateway is generic — it serves any domain.
  What counts as 'toxic' or 'out of domain' is a DOMAIN decision.
    Pharma agent:    blocks self-harm, drug synthesis instructions
    Finance agent:   blocks market manipulation, insider trading queries
    Marketing agent: blocks competitor sabotage requests
  A generic gateway cannot know these rules without coupling to domain logic.

What this handles (domain-specific):
  - Toxic content  (pharma-domain toxic patterns)

What this does NOT handle:
  - Prompt injection → gateway already caught that before agent.run()
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from agent.guardrails import check_toxic

log = logging.getLogger(__name__)


class ContentFilterMiddleware(BaseAgentMiddleware):

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # Check the LAST message — that is the current user input.
        # WHY last not first-human: the new question is always appended last
        # by LangChain before before_agent fires, so last message == current input.
        last_msg = messages[-1]
        if not (hasattr(last_msg, "type") and last_msg.type == "human"):
            return None

        user_content = str(last_msg.content)
        if not user_content.strip():
            return None

        ok, reason = check_toxic(user_content)
        if not ok:
            log.warning(f"[CONTENT_FILTER] Toxic blocked  reason='{reason}'  input='{user_content[:60]}'")
            return {
                "messages": [AIMessage(content=(
                    "Your request could not be processed — it matches a prohibited "
                    f"content pattern for this domain. Reason: {reason}."
                ))],
                "jump_to": "end",
            }

        log.info(f"[CONTENT_FILTER] Passed  input='{user_content[:60]}'")
        return None
