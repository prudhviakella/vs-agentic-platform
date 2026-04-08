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


def get_trace_table_name(app_name: str = "clinical-agent") -> str:
    """
    Fetch the DynamoDB trace table name from SSM Parameter Store.

    SSM path: /{app_name}/{env}/dynamodb/trace_table_name
    Local fallback env var: TRACE_TABLE_NAME
    Local default (if env var also missing): "{app_name}-traces"

    Follows the same pattern as every other config value in this module —
    callers never hardcode table names; they always come from SSM.
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
# This is the single source of truth for local dev.
# MUST be kept in sync with the Bedrock prompt in AWS (README § Bedrock Setup).
#
# THE MANDATORY CLARIFICATION RULE IS NON-NEGOTIABLE.
# Without it, the LLM asks clarifying questions in plain text instead of
# calling ask_user_input. HumanInTheLoopMiddleware never sees a tool call,
# the graph never interrupts, and HITL is silently broken.
# Every time you update the Bedrock prompt in AWS, update this constant too.
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
Failing to call the tool means the user cannot respond interactively.
Use ask_user_input when:
- The trial name or drug is ambiguous
- The question could refer to multiple phases or indications
- The user intent is unclear

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

    Local mode: returns BEDROCK_PROMPT_TEMPLATE env var if set (useful for
    testing prompt changes without AWS), otherwise falls back to
    _LOCAL_SYSTEM_PROMPT which mirrors the production Bedrock prompt exactly.

    AWS mode: fetches the versioned prompt from Bedrock Prompt Management.
    The returned template uses {{double_braces}} for LangChain variables
    (domain_frame, episodic_context, max_tool_calls).
    """
    # Local mode OR sentinel values from prompt.py's local short-circuit.
    # prompt.py calls _fetch_prompt_template("local", "0") to skip SSM —
    # both branches must return _LOCAL_SYSTEM_PROMPT (or BEDROCK_PROMPT_TEMPLATE
    # if the developer set it for manual prompt testing).
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
#
# DynamoDB is the right store for agent traces:
#   • Write-once, read-by-run-id  → simple PK lookup, no joins needed
#   • Schema-flexible             → trace dict maps directly to a DynamoDB item
#   • TTL built-in                → auto-expire old traces at zero cost
#
# Follows the same pattern as _ssm() / _secretsmanager() / _bedrock() above:
# the boto3 resource is cached here; callers (TracerMiddleware) have zero boto3.

@lru_cache(maxsize=None)
def _dynamodb(region: str = "us-east-1") -> Any:
    """
    Cached boto3 DynamoDB *resource* (high-level API).

    We use resource (not client) so table operations like
    table.put_item() and table.get_item() are object-oriented and
    don't require manually constructing AttributeValue dicts.

    lru_cache key includes region so multi-region deployments work correctly.
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

    This function is designed to be called once at startup (or lazily on first
    write). Subsequent calls return the existing table immediately via
    load() — DescribeTable costs ~1 ms and is called at most once per process.

    Table design
    ─────────────
    PK  run_id     (S)  — unique per agent request (LangGraph run_id)
        ts         (N)  — Unix epoch float, for sorting in the DynamoDB console
        expires_at (N)  — Unix epoch int, TTL attribute; DynamoDB auto-deletes
                          items within 48 h of this timestamp
        + all trace fields (question, answer, tools_called, elapsed_ms, …)

    No GSI by default. Add a GSI on (agent, ts) only when you have a real
    query pattern that needs it — GSIs cost money even when idle.

    BillingMode=PAY_PER_REQUEST: no capacity planning needed for trace volumes.
    Switch to PROVISIONED only if you have steady, high-volume load.

    Args:
        table_name: DynamoDB table name to create or confirm.
        ttl_days:   Days before a trace item is auto-expired. 0 = no TTL.
        region:     AWS region for the DynamoDB resource.

    Returns:
        boto3 DynamoDB Table resource, confirmed ACTIVE.
    """
    from botocore.exceptions import ClientError

    ddb   = _dynamodb(region)
    table = ddb.Table(table_name)

    try:
        # load() = DescribeTable. Cheap — confirms table exists and is ACTIVE.
        table.load()
        log.info(f"[AWS] DynamoDB table '{table_name}' found  region={region}")

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise  # Unexpected error (permissions, network) — surface it

        # Table doesn't exist → create it now.
        log.info(f"[AWS] DynamoDB table '{table_name}' not found — creating…")
        table = ddb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        # Block until ACTIVE — create_table is async; wait_until_exists polls.
        table.wait_until_exists()

        if ttl_days > 0:
            # TTL deletes happen automatically within 48 h of expires_at.
            # There is no extra charge for TTL deletions.
            ddb.meta.client.update_time_to_live(
                TableName=table_name,
                TimeToLiveSpecification={
                    "Enabled":       True,
                    "AttributeName": "expires_at",
                },
            )
            log.info(
                f"[AWS] TTL enabled on '{table_name}'.expires_at "
                f"(auto-expire after {ttl_days} days)"
            )

        log.info(f"[AWS] DynamoDB table '{table_name}' created and ACTIVE")

    return table


def put_trace(table: Any, trace: dict, ttl_days: int = 30) -> None:
    """
    Serialize a trace dict and write it to DynamoDB.

    Serialization rules
    ─────────────────────
    • float  → Decimal  DynamoDB rejects Python float; Decimal is required.
    • None   → omitted  DynamoDB rejects null attribute values by default.
    • list/dict fields (tool_results, tools_called) are stored as L / M types
      automatically by the boto3 resource layer — no manual AttributeValue needed.

    Args:
        table:    boto3 DynamoDB Table resource (from init_trace_table).
        trace:    Trace dict produced by TracerMiddleware._extract_from_state.
        ttl_days: Days until the item is auto-expired. 0 = no TTL attribute written.

    Raises:
        Nothing — all exceptions are caught and logged.
        Observability must never break the agent response path.
    """
    from decimal import Decimal

    def _to_decimal(v: Any) -> Any:
        """Recursively convert float → Decimal inside nested dicts/lists."""
        if isinstance(v, float):
            return Decimal(str(round(v, 4)))
        if isinstance(v, dict):
            return {k: _to_decimal(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_to_decimal(i) for i in v]
        return v

    try:
        item = _to_decimal(dict(trace))

        # Guarantee PK is present and is a string.
        item["run_id"] = str(trace.get("run_id", "unknown"))

        # Readable timestamp for DynamoDB console sorting.
        item["ts"] = Decimal(str(round(time.time(), 3)))

        # TTL: epoch seconds N days from now. DynamoDB removes the item after
        # this time, within a 48-hour window (deletions are free).
        if ttl_days > 0:
            item["expires_at"] = int(time.time()) + ttl_days * 86_400

        # Strip None values — boto3 resource layer rejects them without explicit
        # NULL type handling. Omitting is cleaner than sending NULL.
        item = {k: v for k, v in item.items() if v is not None}

        table.put_item(Item=item)
        log.debug(f"[AWS] Trace persisted  run_id={item['run_id']}")

    except Exception as exc:
        log.error(f"[AWS] DynamoDB put_trace failed  run_id={trace.get('run_id')}  error={exc}")


def get_trace_item(table_name: str, run_id: str, region: str = "us-east-1") -> dict | None:
    """
    Fetch a single trace item from DynamoDB by run_id.

    Intended for admin/debug endpoints — not on the hot agent path.

    Args:
        table_name: DynamoDB table name.
        run_id:     Trace primary key.
        region:     AWS region.

    Returns:
        The item dict, or None if not found or on error.
    """
    try:
        table  = _dynamodb(region).Table(table_name)
        result = table.get_item(Key={"run_id": run_id})
        return result.get("Item")
    except Exception as exc:
        log.error(f"[AWS] DynamoDB get_trace failed  run_id={run_id}  error={exc}")
        return None