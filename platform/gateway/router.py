"""
gateway/router.py — Agent API Router
======================================
FastAPI routes for all agent interactions.

Endpoints:
  POST /api/v1/{agent}/chat      — send a message, get a response
  POST /api/v1/{agent}/resume    — resume a paused HITL conversation
  GET  /api/v1/health            — liveness check

Agent routing:
  The {agent} path parameter maps to a registered agent factory.
  New agents are registered in AGENT_REGISTRY — no code changes elsewhere.

  Current registry:
    "clinical-trial" → clinical_trial_agent.agent.agent.build_agent

HITL flow:
  POST /chat   → response.interrupted=True + interrupt_payload
  (user reads question, submits answer)
  POST /resume → response.interrupted=False + final answer

Error handling:
  All agent exceptions are caught and returned as structured ErrorResponse.
  Raw tracebacks never reach the client.
"""

import logging
import time
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Path

from platform.gateway.auth import AuthContext, require_auth
from platform.gateway.injection import check_injection
from platform.gateway.rate_limiter import check_rate_limit
from platform.gateway.schemas import (
    ChatRequest, ChatResponse,
    HITLResumeRequest, ErrorResponse,
)
from platform.observability.tracer import get_current_request_id

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agents"])


# ── Agent registry ─────────────────────────────────────────────────────────────

def _load_clinical_trial_agent(domain: str):
    from agent.agent import build_agent
    return build_agent(domain=domain)


# Maps URL path segment → agent factory function.
# Each factory is called once and the agent is cached per domain.
AGENT_REGISTRY: dict[str, Callable] = {
    "clinical-trial": _load_clinical_trial_agent,
}

# Agent instance cache — one per (agent_name, domain) pair.
_agent_cache: dict[str, Any] = {}


def _get_agent(agent_name: str, domain: str) -> Any:
    """
    Return a cached agent instance, building it on first call.

    WHY cache agents:
      build_agent() fetches credentials from AWS, creates Pinecone clients,
      and sets up a PostgresSaver connection — expensive operations that
      should not repeat on every request.
    """
    key = f"{agent_name}:{domain}"
    if key not in _agent_cache:
        factory = AGENT_REGISTRY.get(agent_name)
        if not factory:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found. Available: {list(AGENT_REGISTRY)}",
            )
        log.info(f"[ROUTER] Building agent  name={agent_name}  domain={domain}")
        _agent_cache[key] = factory(domain)
    return _agent_cache[key]


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    """Liveness check — returns 200 with platform status."""
    return {
        "status":  "ok",
        "agents":  list(AGENT_REGISTRY.keys()),
    }


@router.post(
    "/{agent}/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Prompt injection detected"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Agent error"},
    },
)
async def chat(
    body:  ChatRequest,
    agent: str         = Path(..., description="Agent name, e.g. 'clinical-trial'"),
    auth:  AuthContext = Depends(require_auth),
    _:     None        = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Send a message to the specified agent and get a response.

    Request flow:
      1. Auth validated by require_auth dependency.
      2. Rate limit checked by check_rate_limit dependency.
      3. Prompt injection check runs on the message content.
      4. Agent is retrieved from cache (or built on first call).
      5. Agent invoked — may return a final answer or a HITL interrupt.
      6. Response structured and returned with trace metadata.

    When interrupted=True:
      The agent paused for human input (ask_user_input tool was called).
      The caller must read interrupt_payload.question + options, collect
      the human's answer, and POST to /resume with the same thread_id.
    """
    request_id = get_current_request_id()
    t0         = time.perf_counter()

    # Gateway-level injection check — runs before the agent is invoked.
    check_injection(body.message, request_id=request_id)

    try:
        instance = _get_agent(agent, body.domain)

        messages = [{"role": m.role, "content": m.content} for m in body.history]
        messages.append({"role": "user", "content": body.message})

        config  = {"configurable": {"thread_id": body.thread_id}}
        context = {
            "user_id":    auth.user_id,
            "session_id": body.session_id or body.thread_id,
            "domain":     body.domain,
        }

        response = instance.invoke(
            {"messages": messages},
            config=config,
            context=context,
        )

        elapsed = round((time.perf_counter() - t0) * 1_000, 2)

        # ── HITL interrupt ─────────────────────────────────────────────────
        if response.get("__interrupt__"):
            interrupt_val = response["__interrupt__"][0].value
            action        = interrupt_val["action_requests"][0]
            return ChatResponse(
                answer            = "",
                thread_id         = body.thread_id,
                request_id        = request_id,
                interrupted       = True,
                interrupt_payload = {
                    "question":       action["args"]["question"],
                    "options":        action["args"]["options"],
                    "allow_freetext": action["args"].get("allow_freetext", True),
                },
                agent      = agent,
                latency_ms = elapsed,
            )

        # ── Final answer ───────────────────────────────────────────────────
        messages_out = response.get("messages", [])
        answer = ""
        for msg in reversed(messages_out):
            if hasattr(msg, "content") and msg.content and \
               not getattr(msg, "tool_calls", None):
                answer = str(msg.content)
                break

        return ChatResponse(
            answer     = answer,
            thread_id  = body.thread_id,
            request_id = request_id,
            agent      = agent,
            latency_ms = elapsed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "[ROUTER] Agent error",
            extra={"agent": agent, "request_id": request_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal agent error. Check logs.")


@router.post(
    "/{agent}/resume",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def resume(
    body:  HITLResumeRequest,
    agent: str         = Path(..., description="Agent name"),
    auth:  AuthContext = Depends(require_auth),
    _:     None        = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Resume a paused HITL conversation with the human's answer.

    The agent retrieves the paused graph state from PostgresSaver using
    thread_id, injects the user_answer, and continues execution.
    Returns the final answer once no more interrupts remain.
    """
    from langgraph.types import Command

    request_id = get_current_request_id()
    t0         = time.perf_counter()

    try:
        instance = _get_agent(agent, "pharma")   # domain preserved in checkpoint

        resume_command = Command(
            resume={
                "decisions": [{
                    "type": "edit",
                    "edited_action": {
                        "name": "ask_user_input",
                        "args": {"user_answer": body.user_answer},
                    },
                }]
            }
        )

        config   = {"configurable": {"thread_id": body.thread_id}}
        response = instance.invoke(resume_command, config=config)
        elapsed  = round((time.perf_counter() - t0) * 1_000, 2)

        # Another interrupt — agent asked a follow-up question
        if response.get("__interrupt__"):
            interrupt_val = response["__interrupt__"][0].value
            action        = interrupt_val["action_requests"][0]
            return ChatResponse(
                answer            = "",
                thread_id         = body.thread_id,
                request_id        = request_id,
                interrupted       = True,
                interrupt_payload = {
                    "question": action["args"]["question"],
                    "options":  action["args"]["options"],
                },
                agent      = agent,
                latency_ms = elapsed,
            )

        messages_out = response.get("messages", [])
        answer = ""
        for msg in reversed(messages_out):
            if hasattr(msg, "content") and msg.content and \
               not getattr(msg, "tool_calls", None):
                answer = str(msg.content)
                break

        return ChatResponse(
            answer     = answer,
            thread_id  = body.thread_id,
            request_id = request_id,
            agent      = agent,
            latency_ms = elapsed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "[ROUTER] Resume error",
            extra={"agent": agent, "request_id": request_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal agent error. Check logs.")
