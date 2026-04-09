"""
prompt_versioning/manager.py — Bedrock Prompt Version Manager
==============================================================
Manages prompt versions for all registered agents via Bedrock Prompt
Management + SSM. Provides list, activate, and rollback operations.

How prompt versioning works:
  - The prompt template text lives in Bedrock Prompt Management.
  - The active version pointer lives in SSM:
      /{app_name}/{env}/bedrock/prompt_version  <- e.g. "3"
  - Activating a version = updating that SSM parameter.
  - The agent's prompt.py reads SSM on every request — change takes
    effect on the next request with zero downtime.

Rollback strategy:
  SSM stores only the current active version, not history. The manager
  maintains a simple previous_version key in SSM:
    /{app_name}/{env}/bedrock/prompt_version_previous
  Rollback sets active <- previous and previous <- active.
  Only one level of rollback is supported — for deeper history, use
  the activate endpoint with an explicit version number.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from core import aws

log = logging.getLogger(__name__)

# Maps agent URL slug -> SSM app_name prefix
AGENT_APP_NAMES = {
    "clinical-trial": "clinical-trial-agent",
}


@lru_cache(maxsize=1)
def _bedrock_agent_client():
    """Cached Bedrock Agent client for prompt management operations."""
    return boto3.client("bedrock-agent")


@dataclass
class PromptVersionInfo:
    version:     str
    is_active:   bool
    description: str = ""


def _ssm_path(app_name: str, env: str, key: str) -> str:
    return f"/{app_name}/{env}/bedrock/{key}"


def get_prompt_id(app_name: str, env: str) -> str:
    """Fetch the Bedrock prompt resource ID from SSM."""
    return aws.get_ssm_parameter(_ssm_path(app_name, env, "prompt_id"), with_decryption=False)


def get_active_version(app_name: str, env: str) -> str:
    """Fetch the currently active prompt version from SSM."""
    return aws.get_ssm_parameter(_ssm_path(app_name, env, "prompt_version"), with_decryption=False)


def list_versions(agent_slug: str, env: str) -> list[PromptVersionInfo]:
    """
    List all available Bedrock prompt versions for the given agent.
    Returns versions in descending order (newest first).
    """
    app_name   = _resolve_app_name(agent_slug)
    prompt_id  = get_prompt_id(app_name, env)
    active_ver = get_active_version(app_name, env)

    try:
        resp     = _bedrock_agent_client().list_prompt_versions(promptIdentifier=prompt_id)
        versions = []
        for item in resp.get("promptSummaryList", []):
            ver = str(item.get("version", ""))
            versions.append(PromptVersionInfo(
                version     = ver,
                is_active   = (ver == active_ver),
                description = item.get("description", ""),
            ))

        versions.sort(key=lambda v: int(v.version) if v.version.isdigit() else 0, reverse=True)
        return versions

    except ClientError as exc:
        log.error(f"[PROMPT_MGR] Failed to list versions  prompt_id={prompt_id}  err={exc}")
        raise


def activate_version(agent_slug: str, env: str, version: str, reason: str = "") -> tuple[str, str]:
    """
    Activate a specific prompt version by updating SSM.
    Saves the current active version as 'previous' to enable one-step rollback.
    Returns (previous_version, activated_version).
    """
    app_name = _resolve_app_name(agent_slug)
    previous = get_active_version(app_name, env)

    # Validate version exists before touching SSM
    _validate_version_exists(agent_slug, env, version)

    _put_ssm(app_name, env, "prompt_version_previous", previous)
    _put_ssm(app_name, env, "prompt_version",          version)

    log.info(
        "[PROMPT_MGR] Version activated",
        extra={
            "agent":    agent_slug,
            "env":      env,
            "previous": previous,
            "active":   version,
            "reason":   reason,
        },
    )
    return previous, version


def rollback_version(agent_slug: str, env: str) -> tuple[str, str]:
    """
    Roll back to the previous prompt version.
    Swaps active <-> previous in SSM. Only one level of rollback is supported.
    Returns (rolled_back_from, rolled_back_to).
    """
    app_name = _resolve_app_name(agent_slug)
    current  = get_active_version(app_name, env)
    previous = _get_ssm_optional(app_name, env, "prompt_version_previous")

    if not previous:
        raise ValueError(
            f"No previous version recorded for agent '{agent_slug}' env='{env}'. "
            "Cannot rollback — this may be the first activation."
        )

    _put_ssm(app_name, env, "prompt_version",          previous)
    _put_ssm(app_name, env, "prompt_version_previous", current)

    log.info(
        "[PROMPT_MGR] Rolled back",
        extra={"agent": agent_slug, "env": env, "from": current, "to": previous},
    )
    return current, previous


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_app_name(agent_slug: str) -> str:
    app_name = AGENT_APP_NAMES.get(agent_slug)
    if not app_name:
        raise ValueError(f"Unknown agent '{agent_slug}'. Registered: {list(AGENT_APP_NAMES)}")
    return app_name


def _validate_version_exists(agent_slug: str, env: str, version: str) -> None:
    """
    Validate that a version exists in Bedrock before activating it.
    Takes agent_slug (not app_name) so list_versions can resolve correctly.
    """
    versions = list_versions(agent_slug, env)
    known    = {v.version for v in versions}
    if version not in known:
        raise ValueError(f"Version '{version}' does not exist. Available: {sorted(known)}")


def _put_ssm(app_name: str, env: str, key: str, value: str) -> None:
    """Write a value to SSM using the public boto3 client, not a private helper."""
    path = _ssm_path(app_name, env, key)
    boto3.client("ssm").put_parameter(Name=path, Value=value, Type="String", Overwrite=True)


def _get_ssm_optional(app_name: str, env: str, key: str) -> Optional[str]:
    try:
        return aws.get_ssm_parameter(_ssm_path(app_name, env, key), with_decryption=False)
    except ClientError:
        return None
