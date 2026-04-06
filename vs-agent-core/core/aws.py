"""
aws.py — AWS Credential Helpers
================================
Centralises all AWS SDK calls so the rest of the codebase has zero boto3
imports. agent.py, cache.py, and pinecone_store.py stay cloud-agnostic —
they receive initialised clients, not raw credentials.

LOCAL DEV MODE (APP_ENV=local):
  When APP_ENV=local, all credential functions fall back to environment
  variables / .env.local instead of calling AWS. This lets you develop
  and test without an AWS account, Cognito, or SSM setup.

  Copy .env.local.example to .env.local and fill in your values:
    PINECONE_API_KEY=pcsk-...
    PINECONE_INDEX_NAME=clinical-agent
    OPENAI_API_KEY=sk-...
    POSTGRES_URL=postgresql://user:pass@localhost:5432/clinical_agent
    BEDROCK_PROMPT_TEMPLATE=Your prompt template here

  Start with:
    APP_ENV=local python clinical_trial_agent/run.py

SSM parameter paths (prod/staging/dev):
  /clinical-agent/{env}/pinecone/api_key     ← SecureString
  /clinical-agent/{env}/pinecone/index_name  ← String

Secrets Manager secret:
  Name:  clinical-agent/{env}/postgres
  Value: {"host": "...", "port": "5432", "dbname": "...",
          "username": "...", "password": "..."}

{env} is read from APP_ENV environment variable, defaults to "prod".
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)


# ── Environment ────────────────────────────────────────────────────────────────

def get_env() -> str:
    """
    Return the deployment environment name.

    Values: "local" | "dev" | "staging" | "prod"
    Defaults to "prod" so production needs no explicit setting.
    Use APP_ENV=local for local development without AWS.
    """
    return os.environ.get("APP_ENV", "prod")


def is_local() -> bool:
    """Return True when running in local dev mode (APP_ENV=local)."""
    return get_env() == "local"


def _load_dotenv_local() -> None:
    """
    Load .env.local if it exists and python-dotenv is installed.
    Called once at module import time so env vars are available immediately.
    Silent no-op if the file doesn't exist or dotenv isn't installed.
    """
    try:
        from dotenv import load_dotenv
        env_file = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env.local")
        env_file = os.path.abspath(env_file)
        if os.path.exists(env_file):
            load_dotenv(env_file, override=False)
            log.info(f"[AWS] Loaded .env.local from {env_file}")
    except ImportError:
        pass  # python-dotenv not installed — use os.environ directly


_load_dotenv_local()


# ── Boto3 clients ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _ssm() -> Any:
    """Cached boto3 SSM client. Not used in local mode."""
    import boto3
    return boto3.client("ssm")


@lru_cache(maxsize=None)
def _secretsmanager() -> Any:
    """Cached boto3 Secrets Manager client. Not used in local mode."""
    import boto3
    return boto3.client("secretsmanager")


# ── Low-level fetch functions ──────────────────────────────────────────────────

def get_ssm_parameter(name: str, with_decryption: bool = True) -> str:
    """
    Fetch a single AWS SSM Parameter Store value by full path name.

    In local mode (APP_ENV=local): reads from environment variable derived
    from the SSM path. Example:
      /clinical-agent/local/pinecone/api_key → PINECONE_API_KEY

    Args:
        name:             Full SSM parameter path.
        with_decryption:  True for SecureString parameters.

    Returns:
        The parameter value as a plain string.
    """
    if is_local():
        return _get_local_param(name)
    resp = _ssm().get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]


def _get_local_param(ssm_path: str) -> str:
    """
    Map an SSM path to an environment variable for local dev.

    Mapping rules — last path segment uppercased:
      .../pinecone/api_key     → PINECONE_API_KEY
      .../pinecone/index_name  → PINECONE_INDEX_NAME
      .../bedrock/prompt_id    → BEDROCK_PROMPT_ID
      .../bedrock/prompt_version → BEDROCK_PROMPT_VERSION
      .../cognito/user_pool_id → COGNITO_USER_POOL_ID
      .../cognito/region       → COGNITO_REGION
      .../platform/api_key     → PLATFORM_API_KEY

    Falls back to the full path converted to env var format if no match.
    """
    # Take last two segments: "pinecone/api_key" → "PINECONE_API_KEY"
    parts  = [p for p in ssm_path.strip("/").split("/") if p]
    key    = "_".join(parts[-2:]).upper().replace("-", "_")
    value  = os.environ.get(key)
    if value:
        return value

    # Fallback: full path as env var
    fallback = ssm_path.strip("/").replace("/", "_").replace("-", "_").upper()
    value = os.environ.get(fallback)
    if value:
        return value

    raise EnvironmentError(
        f"[LOCAL MODE] Missing env var '{key}' for SSM path '{ssm_path}'. "
        f"Add it to .env.local or export it."
    )


def get_secret_json(secret_name: str) -> dict:
    """
    Fetch an AWS Secrets Manager secret and return it parsed as a dict.

    In local mode: reads POSTGRES_URL env var and reconstructs the dict.
    """
    if is_local():
        return _get_local_secret(secret_name)
    import boto3
    resp       = _secretsmanager().get_secret_value(SecretId=secret_name)
    secret_str = resp.get("SecretString") or resp.get("SecretBinary", b"").decode()
    return json.loads(secret_str)


def _get_local_secret(secret_name: str) -> dict:
    """
    For local dev, read Postgres credentials from POSTGRES_URL env var.

    Format: postgresql://username:password@host:port/dbname
    """
    postgres_url = os.environ.get("POSTGRES_URL", "")
    if postgres_url:
        # Parse: postgresql://user:pass@host:5432/dbname
        from urllib.parse import urlparse
        p = urlparse(postgres_url)
        return {
            "host":     p.hostname or "localhost",
            "port":     str(p.port or 5432),
            "dbname":   p.path.lstrip("/"),
            "username": p.username or "",
            "password": p.password or "",
        }
    raise EnvironmentError(
        f"[LOCAL MODE] Missing POSTGRES_URL for secret '{secret_name}'. "
        "Add POSTGRES_URL=postgresql://user:pass@host:5432/dbname to .env.local"
    )


# ── High-level resource initialisers ──────────────────────────────────────────

def init_pinecone_index() -> Any:
    """
    Fetch Pinecone credentials and return a connected Pinecone Index.

    Local mode: reads PINECONE_API_KEY + PINECONE_INDEX_NAME from env.
    AWS mode:   reads from SSM Parameter Store.
    """
    from pinecone import Pinecone

    env        = get_env()
    api_key    = get_ssm_parameter(f"/clinical-agent/{env}/pinecone/api_key",    with_decryption=True)
    index_name = get_ssm_parameter(f"/clinical-agent/{env}/pinecone/index_name", with_decryption=False)

    log.info(f"[AWS] Pinecone index='{index_name}'  env={env}")
    return Pinecone(api_key=api_key).Index(index_name)


def init_postgres_url() -> str:
    """
    Fetch Postgres credentials and return a connection URL.

    Local mode: reads POSTGRES_URL directly from env.
    AWS mode:   reads from Secrets Manager.

    Never log the returned string — it contains the password.
    """
    if is_local():
        url = os.environ.get("POSTGRES_URL", "")
        if url:
            return url
        raise EnvironmentError(
            "[LOCAL MODE] Missing POSTGRES_URL. "
            "Add POSTGRES_URL=postgresql://user:pass@host:5432/dbname to .env.local"
        )

    env    = get_env()
    secret = get_secret_json(f"clinical-agent/{env}/postgres")

    host     = secret["host"]
    port     = secret.get("port", "5432")
    dbname   = secret["dbname"]
    username = secret["username"]
    password = secret["password"]

    log.info(f"[AWS] Postgres  host={host}  db={dbname}  env={env}")
    from urllib.parse import quote_plus
    return f"postgresql://{username}:{quote_plus(password)}@{host}:{port}/{dbname}"


# ── Bedrock Prompt Management ──────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _bedrock() -> Any:
    """Cached boto3 Bedrock Agent Runtime client."""
    import boto3
    return boto3.client("bedrock-agent-runtime")


def get_bedrock_prompt(prompt_id: str, prompt_version: str = "1") -> str:
    """
    Fetch a prompt template from AWS Bedrock Prompt Management.

    Local mode: returns BEDROCK_PROMPT_TEMPLATE env var, or a sensible
    inline default so the agent can run without Bedrock configured.
    """
    if is_local():
        template = os.environ.get("BEDROCK_PROMPT_TEMPLATE", "")
        if template:
            return template
        # Sensible inline default for local development
        log.warning("[LOCAL MODE] BEDROCK_PROMPT_TEMPLATE not set — using inline default")
        return (
            "{{domain_frame}}\n\n"
            "Always retrieve evidence before answering. Cite your sources. "
            "Be precise and concise. Maximum {{max_tool_calls}} tool calls per request.\n\n"
            "{{episodic_context}}"
        )

    resp     = _bedrock().get_prompt(promptIdentifier=prompt_id, promptVersion=prompt_version)
    variants = resp.get("variants", [])
    if not variants:
        raise ValueError(f"Bedrock prompt '{prompt_id}' v{prompt_version} has no variants")

    template = variants[0].get("templateConfiguration", {}).get("text", {}).get("text", "")
    if not template:
        raise ValueError(f"Bedrock prompt '{prompt_id}' v{prompt_version} has no text template")

    log.info(f"[AWS] Bedrock prompt fetched  id={prompt_id}  version={prompt_version}")
    return template


def get_bedrock_prompt_from_ssm(app_name: str) -> str:
    """Fetch prompt_id and prompt_version from SSM then call get_bedrock_prompt()."""
    env            = get_env()
    prompt_id      = get_ssm_parameter(f"/{app_name}/{env}/bedrock/prompt_id",      with_decryption=False)
    prompt_version = get_ssm_parameter(f"/{app_name}/{env}/bedrock/prompt_version",  with_decryption=False)
    return get_bedrock_prompt(prompt_id, prompt_version)