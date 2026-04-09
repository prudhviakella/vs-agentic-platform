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

  Start with:
    APP_ENV=local python clinical_trial_agent/run.py

SSM parameter paths (prod/staging/dev):
  /clinical-agent/{env}/pinecone/api_key     <- SecureString
  /clinical-agent/{env}/pinecone/index_name  <- String

Secrets Manager secret:
  Name:  clinical-agent/{env}/postgres
  Value: {"host": "...", "port": "5432", "dbname": "...",
          "username": "...", "password": "..."}

{env} is read from APP_ENV environment variable, defaults to "prod".
"""

import json
import logging
import os
import time
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
    return os.environ.get("APP_ENV", "dev")


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
        pass


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

    Local mode: reads from environment variable derived from the SSM path.
    Example: /clinical-agent/local/pinecone/api_key -> PINECONE_API_KEY
    """
    if is_local():
        return _get_local_param(name)
    resp = _ssm().get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]


def _get_local_param(ssm_path: str) -> str:
    """Map an SSM path to an environment variable for local dev."""
    parts  = [p for p in ssm_path.strip("/").split("/") if p]
    key    = "_".join(parts[-2:]).upper().replace("-", "_")
    value  = os.environ.get(key)
    if value:
        return value

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
    Local mode: reads POSTGRES_URL env var and reconstructs the dict.
    """
    if is_local():
        return _get_local_secret(secret_name)
    resp       = _secretsmanager().get_secret_value(SecretId=secret_name)
    secret_str = resp.get("SecretString") or resp.get("SecretBinary", b"").decode()
    return json.loads(secret_str)


def _get_local_secret(secret_name: str) -> dict:
    """For local dev, read Postgres credentials from POSTGRES_URL env var."""
    postgres_url = os.environ.get("POSTGRES_URL", "")
    if postgres_url:
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


def get_trace_table_name(app_name: str = "clinical-agent") -> str:
    """
    Fetch the DynamoDB trace table name from SSM Parameter Store.
    Local fallback: TRACE_TABLE_NAME env var or "{app_name}-traces".
    """
    if is_local():
        return os.environ.get("TRACE_TABLE_NAME", f"{app_name}-traces")
    env = get_env()
    return get_ssm_parameter(
        f"/{app_name}/{env}/dynamodb/trace_table_name",
        with_decryption=False,
    )


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


# ── LOCAL DEFAULT SYSTEM PROMPT ────────────────────────────────────────────────
# Single source of truth for local dev.
# Must be kept in sync with the Bedrock prompt in AWS (README § Bedrock Setup).
#
# MANDATORY CLARIFICATION RULE — ONE QUESTION MAXIMUM:
# The LLM tends to ask multiple rounds of clarifying questions when the rule
# does not explicitly limit it. "After receiving one clarification, proceed"
# is required to prevent infinite HITL loops.
# ──────────────────────────────────────────────────────────────────────────────
_LOCAL_SYSTEM_PROMPT = """\
{domain_frame}

You are an expert clinical research assistant with deep knowledge of \
pharmaceutical drug development, clinical trial design, regulatory \
frameworks (FDA, EMA, ICH), and evidence-based medicine.

CORE BEHAVIOUR:
- Always retrieve evidence before answering. Never answer from memory alone.
- Cite the specific source, trial name, or document for every clinical claim.
- If the retrieved evidence is insufficient, say so explicitly.
- Be precise with numbers — dosages, p-values, endpoints, sample sizes matter.

CLARIFICATION RULE — MANDATORY:
When the request is ambiguous, you MUST call the ask_user_input tool.
Do NOT ask clarifying questions in plain text — ALWAYS use the tool.
You may ask AT MOST ONE clarifying question per request.
After receiving the user's answer, you MUST immediately proceed to search
and answer — do NOT ask follow-up clarifying questions.
Use ask_user_input when:
- The trial name or drug is ambiguous
- The question could refer to multiple phases or indications
- The user intent is completely unclear

DISCLAIMERS:
- Always include: This information is for research purposes only and does not constitute medical advice.
- Never recommend specific treatments for individual patients.
- Flag if data is preliminary, unpublished, or from a single study.

TOOL USAGE:
- Maximum {max_tool_calls} tool calls per request.
- Use search for recent trials and regulatory decisions.
- Use graph for relationships between drugs, targets, and indications.
- Use summariser for long documents.
- Use chart only when visualising data adds clarity.

{episodic_context}\
"""


def get_bedrock_prompt(prompt_id: str, prompt_version: str = "1") -> str:
    """
    Fetch a prompt template from AWS Bedrock Prompt Management.

    Local mode: returns BEDROCK_PROMPT_TEMPLATE env var if set,
    otherwise falls back to _LOCAL_SYSTEM_PROMPT.

    AWS mode: fetches the versioned prompt from Bedrock Prompt Management.
    """
    if is_local() or prompt_id == "local":
        template = os.environ.get("BEDROCK_PROMPT_TEMPLATE", "")
        if template:
            log.info("[LOCAL MODE] Using BEDROCK_PROMPT_TEMPLATE from env")
            return template
        log.warning(
            "[LOCAL MODE] BEDROCK_PROMPT_TEMPLATE not set — "
            "using _LOCAL_SYSTEM_PROMPT (mirrors Bedrock prod prompt)"
        )
        return _LOCAL_SYSTEM_PROMPT

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


# ── DynamoDB — Trace Persistence ───────────────────────────────────────────────

@lru_cache(maxsize=None)
def _dynamodb(region: str = "us-east-1") -> Any:
    """
    Cached boto3 DynamoDB resource (high-level API).
    Using resource (not client) so table.put_item() doesn't require
    manually constructing AttributeValue dicts.
    """
    import boto3
    return boto3.resource("dynamodb", region_name=region)


def init_trace_table(
    table_name: str,
    ttl_days:   int = 30,
    region:     str = "us-east-1",
) -> Any:
    """
    Return a DynamoDB Table resource, creating the table if it does not exist.

    Table design:
      PK  run_id     (S)  — unique per agent request
          ts         (N)  — Unix epoch float for console sorting
          expires_at (N)  — TTL attribute, auto-deleted after ttl_days

    BillingMode=PAY_PER_REQUEST: no capacity planning needed.
    """
    from botocore.exceptions import ClientError

    ddb   = _dynamodb(region)
    table = ddb.Table(table_name)

    try:
        table.load()
        log.info(f"[AWS] DynamoDB table '{table_name}' found  region={region}")

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

        log.info(f"[AWS] DynamoDB table '{table_name}' not found — creating...")
        table = ddb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()

        if ttl_days > 0:
            ddb.meta.client.update_time_to_live(
                TableName=table_name,
                TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
            )
            log.info(f"[AWS] TTL enabled on '{table_name}'.expires_at ({ttl_days} days)")

        log.info(f"[AWS] DynamoDB table '{table_name}' created and ACTIVE")

    return table


def put_trace(table: Any, trace: dict, ttl_days: int = 30) -> None:
    """
    Serialize a trace dict and write it to DynamoDB.
    float -> Decimal (DynamoDB rejects Python float).
    None values are stripped (DynamoDB rejects null attributes by default).
    Errors are caught and logged — observability must never break agent responses.
    """
    from decimal import Decimal

    def _to_decimal(v: Any) -> Any:
        if isinstance(v, float):
            return Decimal(str(round(v, 4)))
        if isinstance(v, dict):
            return {k: _to_decimal(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_to_decimal(i) for i in v]
        return v

    try:
        item = _to_decimal(dict(trace))
        item["run_id"]     = str(trace.get("run_id", "unknown"))
        item["ts"]         = Decimal(str(round(time.time(), 3)))

        if ttl_days > 0:
            item["expires_at"] = int(time.time()) + ttl_days * 86_400

        item = {k: v for k, v in item.items() if v is not None}
        table.put_item(Item=item)
        log.debug(f"[AWS] Trace persisted  run_id={item['run_id']}")

    except Exception as exc:
        log.error(f"[AWS] DynamoDB put_trace failed  run_id={trace.get('run_id')}  error={exc}")


def get_trace_item(table_name: str, run_id: str, region: str = "us-east-1") -> dict | None:
    """
    Fetch a single trace item from DynamoDB by run_id.
    Intended for admin/debug use — not on the hot agent path.
    """
    try:
        table  = _dynamodb(region).Table(table_name)
        result = table.get_item(Key={"run_id": run_id})
        return result.get("Item")
    except Exception as exc:
        log.error(f"[AWS] DynamoDB get_trace failed  run_id={run_id}  error={exc}")
        return None