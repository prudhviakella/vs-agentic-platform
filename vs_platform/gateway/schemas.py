"""
gateway/schemas.py — Pydantic Request / Response Models
=========================================================
All FastAPI endpoint models live here. Keeping them in one file means
the OpenAPI schema is generated from a single source of truth.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Chat request / response ────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role:    str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    """
    POST /api/v1/{agent}/chat

    The thread_id groups messages into a conversation. Callers must supply
    the same thread_id across turns to maintain HITL checkpoint continuity.
    session_id is optional metadata for audit logging.

    history is accepted but not forwarded to the agent — PostgresSaver
    restores prior message history from the checkpoint using thread_id.
    Passing history again would duplicate messages in state.
    """
    message:    str               = Field(...,         description="Current user message")
    thread_id:  str               = Field(...,         description="Conversation thread ID — stable across turns")
    session_id: Optional[str]     = Field(None,        description="Optional session identifier for audit")
    domain:     str               = Field("pharma",    description="Agent domain: 'pharma' | 'general'")
    history:    list[ChatMessage] = Field(default_factory=list, description="Prior turns — not forwarded to agent (checkpointer handles history)")


class ChatResponse(BaseModel):
    """
    Successful chat response.
    interrupted=True means the agent paused for HITL — caller must resume.
    """
    answer:            str   = Field(...,   description="Agent's response text")
    thread_id:         str   = Field(...,   description="Echo of request thread_id")
    request_id:        str   = Field(...,   description="Platform-assigned trace ID")
    interrupted:       bool  = Field(False, description="True if agent paused for human input")
    interrupt_payload: Any   = Field(None,  description="HITL question + options when interrupted=True")
    agent:             str   = Field(...,   description="Agent that handled the request")
    latency_ms:        float = Field(...,   description="End-to-end request latency in milliseconds")


class HITLResumeRequest(BaseModel):
    """
    POST /api/v1/{agent}/resume

    Resume a paused HITL conversation. user_answer is the human's
    response to the ask_user_input question. domain must match the
    original chat request so the correct agent instance is retrieved.
    """
    thread_id:   str = Field(...,      description="Thread ID of the paused conversation")
    user_answer: str = Field(...,      description="Human's answer to the HITL question")
    domain:      str = Field("pharma", description="Agent domain — must match the original chat request")


# ── Error response ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""
    error:      str           = Field(...,  description="Error type / code")
    message:    str           = Field(...,  description="Human-readable error description")
    request_id: Optional[str] = Field(None, description="Trace ID for log correlation")


# ── Prompt versioning ──────────────────────────────────────────────────────────

class PromptVersion(BaseModel):
    """Metadata for a single Bedrock prompt version."""
    version:     str  = Field(..., description="Bedrock prompt version string")
    is_active:   bool = Field(..., description="Whether this is the currently active version")
    description: str  = Field("",  description="Optional description from Bedrock")


class PromptVersionListResponse(BaseModel):
    """Response for GET /api/v1/prompts/{agent}/{env}"""
    agent:          str                 = Field(..., description="Agent name")
    env:            str                 = Field(..., description="Deployment environment")
    prompt_id:      str                 = Field(..., description="Bedrock prompt resource ID")
    active_version: str                 = Field(..., description="Currently active version")
    versions:       list[PromptVersion] = Field(..., description="All available versions")


class PromptActivateRequest(BaseModel):
    """POST /api/v1/prompts/{agent}/{env}/activate"""
    version: str = Field(..., description="Version to activate")
    reason:  str = Field("",  description="Optional reason for audit log")


class PromptActivateResponse(BaseModel):
    """Response after activating or rolling back a prompt version."""
    agent:      str = Field(..., description="Agent name")
    env:        str = Field(..., description="Deployment environment")
    previous:   str = Field(..., description="Version that was active before")
    activated:  str = Field(..., description="Version now active")
    request_id: str = Field(..., description="Trace ID")
