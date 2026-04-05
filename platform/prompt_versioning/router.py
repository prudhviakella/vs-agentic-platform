"""
prompt_versioning/router.py — Prompt Management API
======================================================
FastAPI routes for managing Bedrock prompt versions.

Endpoints:
  GET  /api/v1/prompts/{agent}/{env}              — list all versions
  GET  /api/v1/prompts/{agent}/{env}/active       — get active version
  POST /api/v1/prompts/{agent}/{env}/activate     — activate a version
  POST /api/v1/prompts/{agent}/{env}/rollback     — rollback to previous

Access control:
  All prompt management endpoints require admin scope.
  Only API key callers (internal services / CI pipelines) or Cognito
  users with the 'admin' scope can call these endpoints.

WHY separate router (not inline in gateway/router.py):
  Prompt management is a platform concern, not an agent concern.
  Keeping it separate makes it easy to mount/unmount or restrict to
  internal VPC-only routing without touching the agent API surface.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Path

from platform.gateway.auth import AuthContext, require_auth
from platform.gateway.schemas import (
    PromptVersionListResponse,
    PromptActivateRequest,
    PromptActivateResponse,
    PromptVersion,
)
from platform.observability.tracer import get_current_request_id
from platform.prompt_versioning import manager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["prompt-management"])


def _require_admin(auth: AuthContext = Depends(require_auth)) -> AuthContext:
    """
    Sub-dependency that enforces admin scope on prompt management endpoints.
    Raises 403 if the caller is not admin.
    """
    if "admin" not in auth.scopes and auth.auth_mode != "api_key":
        raise HTTPException(
            status_code=403,
            detail="Admin scope required for prompt management operations.",
        )
    return auth


@router.get("/{agent}/{env}", response_model=PromptVersionListResponse)
async def list_versions(
    agent: str         = Path(..., description="Agent slug, e.g. 'clinical-trial'"),
    env:   str         = Path(..., description="Environment: prod | staging | dev"),
    auth:  AuthContext = Depends(_require_admin),
) -> PromptVersionListResponse:
    """List all available prompt versions for an agent in a given environment."""
    try:
        app_name   = manager._resolve_app_name(agent)
        prompt_id  = manager.get_prompt_id(app_name, env)
        active_ver = manager.get_active_version(app_name, env)
        versions   = manager.list_versions(agent, env)

        return PromptVersionListResponse(
            agent          = agent,
            env            = env,
            prompt_id      = prompt_id,
            active_version = active_ver,
            versions       = [
                PromptVersion(
                    version   = v.version,
                    is_active = v.is_active,
                    description = v.description,
                )
                for v in versions
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.error(f"[PROMPT_ROUTER] list_versions error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list prompt versions")


@router.get("/{agent}/{env}/active")
async def get_active_version(
    agent: str         = Path(...),
    env:   str         = Path(...),
    auth:  AuthContext = Depends(_require_admin),
) -> dict:
    """Return the currently active prompt version for an agent."""
    try:
        app_name = manager._resolve_app_name(agent)
        version  = manager.get_active_version(app_name, env)
        return {"agent": agent, "env": env, "active_version": version}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{agent}/{env}/activate", response_model=PromptActivateResponse)
async def activate_version(
    body:  PromptActivateRequest,
    agent: str         = Path(...),
    env:   str         = Path(...),
    auth:  AuthContext = Depends(_require_admin),
) -> PromptActivateResponse:
    """
    Activate a specific prompt version.

    Takes effect on the next agent request — no restart required.
    The previous version is saved to SSM to enable one-step rollback.
    """
    request_id = get_current_request_id()
    try:
        previous, activated = manager.activate_version(agent, env, body.version, body.reason)
        return PromptActivateResponse(
            agent      = agent,
            env        = env,
            previous   = previous,
            activated  = activated,
            request_id = request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error(f"[PROMPT_ROUTER] activate error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to activate prompt version")


@router.post("/{agent}/{env}/rollback", response_model=PromptActivateResponse)
async def rollback_version(
    agent: str         = Path(...),
    env:   str         = Path(...),
    auth:  AuthContext = Depends(_require_admin),
) -> PromptActivateResponse:
    """
    Roll back to the previous prompt version.

    Only one level of rollback is supported. For deeper history, use
    the activate endpoint with an explicit version number.
    """
    request_id = get_current_request_id()
    try:
        rolled_from, rolled_to = manager.rollback_version(agent, env)
        return PromptActivateResponse(
            agent      = agent,
            env        = env,
            previous   = rolled_from,
            activated  = rolled_to,
            request_id = request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error(f"[PROMPT_ROUTER] rollback error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to rollback prompt version")
