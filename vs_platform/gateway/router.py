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
  New agents are added to AGENT_REGISTRY — no code changes elsewhere.

  Current registry:
    "clinical-trial" -> clinical_trial_agent.agent.agent.build_agent

HITL flow:
  POST /chat   -> response.interrupted=True + interrupt_payload
  (user reads question, submits answer)
  POST /resume -> response.interrupted=False + final answer

Error handling:
  All agent exceptions are caught and returned as structured ErrorResponse.
  Raw tracebacks never reach the client.
"""

import logging
import time
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Path

from vs_platform.gateway.auth import AuthContext, require_auth
from vs_platform.gateway.injection import check_injection
from vs_platform.gateway.rate_limiter import check_rate_limit
from vs_platform.gateway.schemas import (
    ChatRequest, ChatResponse,
    HITLResumeRequest, ErrorResponse,
)
from vs_platform.observability.tracer import get_current_request_id

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agents"])


# ── Agent registry ─────────────────────────────────────────────────────────────

def _load_clinical_trial_agent(domain: str):
    from agent.agent import build_agent
    return build_agent(domain=domain, use_postgres=True)


# Maps URL path segment -> agent factory function.
# Each factory is called once and the result is cached per (agent, domain).
AGENT_REGISTRY: dict[str, Callable] = {
    "clinical-trial": _load_clinical_trial_agent,
}

# One agent instance per (agent_name, domain) -- avoids rebuilding on every request.
# build_agent() fetches AWS credentials, creates Pinecone clients, and opens a
# PostgresSaver connection -- expensive operations that must not repeat per request.
_agent_cache: dict[str, Any] = {}


def _get_agent(agent_name: str, domain: str) -> Any:
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


def _extract_answer(messages: list) -> str:
    """
    Extract the last AI answer from the message list.

    Skips:
      - Messages with no content
      - AIMessages that contain tool_calls (these are tool invocations, not answers)
      - ToolMessages (type == "tool") — these are tool results, not answers
      - HumanMessages (type == "human") — these are user inputs

    Only returns content from AIMessages that have text content and no
    pending tool calls — i.e. the final answer turn.
    """
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not content:
            continue
        # Skip tool call invocations (AI decided to call a tool)
        if getattr(msg, "tool_calls", None):
            continue
        # Skip tool results and human messages
        msg_type = getattr(msg, "type", "")
        if msg_type in ("tool", "human"):
            continue
        return str(content)
    return ""


def _build_interrupt_payload(response: dict) -> dict:
    """Extract the HITL interrupt payload from a LangGraph interrupt response."""
    action = response["__interrupt__"][0].value["action_requests"][0]
    return {
        "question":       action["args"]["question"],
        "options":        action["args"]["options"],
        "allow_freetext": action["args"].get("allow_freetext", True),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "agents": list(AGENT_REGISTRY.keys())}


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

    Only the current message is passed to invoke() -- not the full history.
    PostgresSaver restores prior message history from the checkpoint using
    thread_id, so passing history again would duplicate messages in state.

    When interrupted=True the agent paused for human input (ask_user_input
    was called). POST to /resume with the same thread_id to continue.
    """
    request_id = get_current_request_id()
    t0         = time.perf_counter()

    try:
        check_injection(body.message, request_id=request_id)

        instance = _get_agent(agent, body.domain)

        config  = {"configurable": {"thread_id": body.thread_id}}
        context = {
            "user_id":    auth.user_id,
            "session_id": body.session_id or body.thread_id,
            "domain":     body.domain,
        }

        # Only pass the new message -- PostgresSaver restores prior history
        response = instance.invoke(
            {"messages": [{"role": "user", "content": body.message}]},
            config=config,
            context=context,
        )

        elapsed = round((time.perf_counter() - t0) * 1_000, 2)

        if response.get("__interrupt__"):
            return ChatResponse(
                answer            = "",
                thread_id         = body.thread_id,
                request_id        = request_id,
                interrupted       = True,
                interrupt_payload = _build_interrupt_payload(response),
                agent             = agent,
                latency_ms        = elapsed,
            )

        return ChatResponse(
            answer     = _extract_answer(response.get("messages", [])),
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
    thread_id, injects user_answer, and continues from where it stopped.
    Returns the final answer once no more interrupts remain.
    """
    from langgraph.types import Command

    request_id = get_current_request_id()
    t0         = time.perf_counter()

    try:
        instance = _get_agent(agent, body.domain)

        config  = {"configurable": {"thread_id": body.thread_id}}
        context = {
            "user_id":    auth.user_id,
            "session_id": body.thread_id,
            "domain":     body.domain,
        }

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

        response = instance.invoke(resume_command, config=config, context=context)
        elapsed  = round((time.perf_counter() - t0) * 1_000, 2)

        # Agent asked a follow-up question
        if response.get("__interrupt__"):
            return ChatResponse(
                answer            = "",
                thread_id         = body.thread_id,
                request_id        = request_id,
                interrupted       = True,
                interrupt_payload = _build_interrupt_payload(response),
                agent             = agent,
                latency_ms        = elapsed,
            )

        return ChatResponse(
            answer     = _extract_answer(response.get("messages", [])),
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