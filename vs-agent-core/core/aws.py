"""
aws.py — AWS Credential Helpers
================================
Centralises all AWS SDK calls so the rest of the codebase has zero boto3
imports. agent.py, cache.py, and pinecone_store.py stay cloud-agnostic —
they receive initialised clients, not raw credentials.

WHY a dedicated module (not inline in agent.py):
  agent.py is assembly code — it wires pillars together.
  Credential fetching is infrastructure code — it talks to AWS.
  Mixing them couples the agent lifecycle to AWS SDK behaviour,
  making it harder to test agent assembly without mocking boto3.
  With this module, tests mock agent.aws and agent.py stays clean.

WHY SSM for Pinecone, Secrets Manager for Postgres:
  Pinecone API key → SSM Parameter Store (SecureString, KMS-encrypted).
    SSM is sufficient for API keys that rotate infrequently and manually.
    The plaintext never appears in ECS task definitions or CloudTrail logs.

  Postgres credentials → Secrets Manager.
    Secrets Manager supports automatic rotation on a schedule — the RDS
    secret is rotated every 30 days without manual intervention via the
    built-in RDS rotation Lambda. SSM has no equivalent native rotation
    for database credentials. Secrets Manager also tracks GetSecretValue
    calls in CloudTrail, providing a full access audit trail.

WHY lru_cache on clients:
  boto3.client() opens a new TLS connection each call. lru_cache ensures
  one client per process — the connection is reused across every credential
  fetch during the application lifetime.

IAM permissions required (ECS task role / EC2 instance profile):
  ssm:GetParameter          on arn:aws:ssm:{region}:{account}:parameter/clinical-agent/{env}/*
  secretsmanager:GetSecretValue on arn:aws:secretsmanager:{region}:{account}:secret:clinical-agent/{env}/*
  kms:Decrypt               on the KMS key used for SecureString parameters

SSM parameter paths:
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

import boto3

log = logging.getLogger(__name__)


# ── Environment ────────────────────────────────────────────────────────────────

def get_env() -> str:
    """
    Return the deployment environment name.

    Reads APP_ENV environment variable — the only env var this module uses.
    Defaults to "prod" so production deployments require no explicit setting.
    Override with APP_ENV=staging|dev for non-prod environments.

    This is intentionally the only os.environ call in the AWS module — all
    credentials come from AWS, not from the environment.
    """
    return os.environ.get("APP_ENV", "prod")


# ── Boto3 clients ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _ssm() -> Any:
    """
    Return a cached boto3 SSM client.

    lru_cache(maxsize=None) stores one client per process lifetime.
    Avoids repeated TLS handshakes and session setup on every parameter fetch.
    The client is constructed lazily on first call, not at import time, so
    tests that mock boto3.client() can patch before the cache is populated.
    """
    return boto3.client("ssm")


@lru_cache(maxsize=None)
def _secretsmanager() -> Any:
    """
    Return a cached boto3 Secrets Manager client.
    Same lazy-init and caching rationale as _ssm().
    """
    return boto3.client("secretsmanager")


# ── Low-level fetch functions ──────────────────────────────────────────────────

def get_ssm_parameter(name: str, with_decryption: bool = True) -> str:
    """
    Fetch a single AWS SSM Parameter Store value by full path name.

    Args:
        name:             Full SSM parameter path, e.g.
                          "/clinical-agent/prod/pinecone/api_key"
        with_decryption:  True for SecureString parameters (API keys, tokens).
                          False for plain String parameters (index names, config).

    Returns:
        The parameter value as a plain string.

    Raises:
        botocore.exceptions.ClientError:
          ParameterNotFound — path does not exist in SSM.
          AccessDeniedException — IAM role lacks ssm:GetParameter on this path.
    """
    resp = _ssm().get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]


def get_secret_json(secret_name: str) -> dict:
    """
    Fetch an AWS Secrets Manager secret and return it parsed as a dict.

    Handles both SecretString (JSON text) and SecretBinary (base64-encoded
    JSON bytes) response formats — SecretBinary is used when the secret was
    created with binary encoding, which some RDS rotation Lambdas produce.

    Args:
        secret_name: The secret name or ARN, e.g.
                     "clinical-agent/prod/postgres"

    Returns:
        Parsed JSON dict of the secret value.

    Raises:
        botocore.exceptions.ClientError:
          ResourceNotFoundException — secret does not exist.
          AccessDeniedException — IAM role lacks secretsmanager:GetSecretValue.
        json.JSONDecodeError:
          Secret exists but is not valid JSON — signals a misconfigured secret,
          not a code bug. The raw value is intentionally not included in the
          exception message to avoid logging credential fragments.
    """
    resp       = _secretsmanager().get_secret_value(SecretId=secret_name)
    secret_str = resp.get("SecretString") or resp.get("SecretBinary", b"").decode()
    return json.loads(secret_str)


# ── High-level resource initialisers ──────────────────────────────────────────

def init_pinecone_index() -> Any:
    """
    Fetch Pinecone credentials from SSM and return a connected Pinecone Index.

    SSM parameters fetched:
      /clinical-agent/{env}/pinecone/api_key     — SecureString (KMS-encrypted)
      /clinical-agent/{env}/pinecone/index_name  — String

    The Pinecone index must already exist with:
      metric:    cosine   (required for SemanticCache and PineconeStore)
      dimension: 1536     (matches text-embedding-3-small output)

    Raises:
      botocore.exceptions.ClientError — SSM access denied or parameter missing.
      pinecone.exceptions.PineconeException — index not found or wrong region.
    """
    from pinecone import Pinecone

    env        = get_env()
    api_key    = get_ssm_parameter(f"/clinical-agent/{env}/pinecone/api_key",    with_decryption=True)
    index_name = get_ssm_parameter(f"/clinical-agent/{env}/pinecone/index_name", with_decryption=False)

    log.info(f"[AWS] Pinecone index='{index_name}'  env={env}")
    return Pinecone(api_key=api_key).Index(index_name)


def init_postgres_url() -> str:
    """
    Fetch Postgres credentials from Secrets Manager and return a connection URL.

    Secret name: clinical-agent/{env}/postgres
    Expected JSON keys: host, port (optional, defaults to 5432),
                        dbname, username, password

    WHY return a URL string (not a PostgresSaver):
      The checkpointer setup (PostgresSaver.from_conn_string + .setup()) is the
      caller's responsibility — it belongs in agent.py where the full agent
      lifecycle is managed. This function is purely a credential resolver.

    WHY construct the URL here (not store it pre-constructed in Secrets Manager):
      A pre-constructed URL containing the password stored as a single secret
      string would expose the password if the string were ever logged. Keeping
      the password as a separate JSON field means it is never concatenated into
      any log-safe variable — only into the connection URL string at return time,
      which is immediately consumed by the caller and never stored.

    Returns:
        PostgreSQL DSN string: "postgresql://user:password@host:port/dbname"
        Never log this string.

    Raises:
      botocore.exceptions.ClientError — Secrets Manager access denied or missing.
      json.JSONDecodeError — secret is not valid JSON (misconfigured secret).
      KeyError — required key missing from the secret JSON.
    """
    env    = get_env()
    secret = get_secret_json(f"clinical-agent/{env}/postgres")

    host     = secret["host"]
    port     = secret.get("port", "5432")
    dbname   = secret["dbname"]
    username = secret["username"]
    password = secret["password"]

    log.info(f"[AWS] Postgres  host={host}  db={dbname}  env={env}")

    # Construct DSN at the last possible moment — caller consumes immediately.
    # Never assign this to a variable that could appear in a log or traceback.
    return f"postgresql://{username}:{password}@{host}:{port}/{dbname}"


# ── Bedrock Prompt Management ──────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _bedrock() -> Any:
    """
    Cached boto3 Bedrock Agent Runtime client for prompt retrieval.
    Uses bedrock-agent-runtime (not bedrock-runtime) — prompt management
    is part of the Agents API surface, not the core inference surface.
    IAM: requires bedrock:GetPrompt on the prompt ARN.
    """
    return boto3.client("bedrock-agent-runtime")


def get_bedrock_prompt(prompt_id: str, prompt_version: str = "1") -> str:
    """
    Fetch a prompt template from AWS Bedrock Prompt Management.

    Prompt ID and version are stored in SSM so they can be updated without
    a code deployment:
      /clinical-trial-agent/{env}/bedrock/prompt_id
      /clinical-trial-agent/{env}/bedrock/prompt_version

    The prompt template is stored in Bedrock with named variables:
      {{domain_frame}}       — injected by the consuming agent's prompt.py
      {{episodic_context}}   — injected from InMemoryStore / PineconeStore
      {{max_tool_calls}}     — injected from tools/__init__.py constant

    WHY Bedrock Prompt Management (not hardcoded strings):
      Prompts are content, not code. Non-engineers (clinical writers, domain
      experts) can edit, version, and A/B test prompts in the Bedrock console
      without touching the codebase or triggering a deployment. Version IDs
      allow instant rollback if a prompt change degrades answer quality.

    Args:
        prompt_id:      Bedrock prompt resource ID (not ARN).
        prompt_version: Prompt version string, defaults to "1".

    Returns:
        The raw prompt template string with {{variable}} placeholders intact.
        The caller is responsible for substituting variables before sending
        the prompt to the LLM.

    Raises:
        botocore.exceptions.ClientError — access denied or prompt not found.
    """
    resp = _bedrock().get_prompt(
        promptIdentifier=prompt_id,
        promptVersion=prompt_version,
    )
    # Bedrock returns variants — take the first text variant's template content.
    variants = resp.get("variants", [])
    if not variants:
        raise ValueError(f"Bedrock prompt '{prompt_id}' v{prompt_version} has no variants")

    template = variants[0].get("templateConfiguration", {}).get("text", {}).get("text", "")
    if not template:
        raise ValueError(f"Bedrock prompt '{prompt_id}' v{prompt_version} has no text template")

    log.info(f"[AWS] Bedrock prompt fetched  id={prompt_id}  version={prompt_version}")
    return template


def get_bedrock_prompt_from_ssm(app_name: str) -> str:
    """
    Convenience wrapper — fetches prompt_id and prompt_version from SSM,
    then calls get_bedrock_prompt().

    SSM parameters:
      /{app_name}/{env}/bedrock/prompt_id
      /{app_name}/{env}/bedrock/prompt_version

    Args:
        app_name: Application name prefix matching the SSM path convention,
                  e.g. "clinical-trial-agent".

    Returns:
        Raw prompt template string from Bedrock.
    """
    env            = get_env()
    prompt_id      = get_ssm_parameter(f"/{app_name}/{env}/bedrock/prompt_id",      with_decryption=False)
    prompt_version = get_ssm_parameter(f"/{app_name}/{env}/bedrock/prompt_version",  with_decryption=False)
    return get_bedrock_prompt(prompt_id, prompt_version)
