# VS Agentic Platform

Multi-agent AI platform built for Vidya Sankalp — Applied GenAI Engineering.

---

## Structure

```
vs-agentic-platform/
  vs-agent-core/          ← shared base package (AWS, Pinecone, middleware)
  clinical_trial_agent/   ← pharma domain agent
  vs_platform/            ← FastAPI gateway (auth, rate limiting, prompt versioning)
  requirements-dev.txt
```

---

## Setup

```bash
# 1. Create venv
python3.13 -m venv .venv && source .venv/bin/activate

# 2. Install all packages
pip install --no-cache-dir -r requirements-dev.txt

# 3. Export environment variables
export OPENAI_API_KEY="sk-..."
export APP_ENV="dev"
```

---

## AWS Prerequisites

Complete these steps in order before running the agent.

### 1. Pinecone Index

Create via Python:

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="your-pinecone-api-key")
pc.create_index(
    name="clinical-agent",
    dimension=1536,       # matches text-embedding-3-small
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

Or create from [app.pinecone.io](https://app.pinecone.io):
- Name: `clinical-agent`, Dimensions: `1536`, Metric: `cosine`
- Cloud: `AWS`, Region: `us-east-1`

Wait ~30 seconds for the index to initialize before running the agent.

### 2. Bedrock System Prompt

Always use a JSON file to avoid shell quoting issues.

**First time — create the prompt:**

```bash
cat > /tmp/prompt_payload.json << 'ENDJSON'
{
  "name": "clinical-trial-agent-system-prompt-1",
  "description": "System prompt for the clinical trial agent",
  "variants": [{
    "name": "default",
    "templateType": "TEXT",
    "templateConfiguration": {
      "text": {
        "text": "{{domain_frame}}\n\nYou are an expert clinical research assistant with deep knowledge of pharmaceutical drug development, clinical trial design, regulatory frameworks (FDA, EMA, ICH), and evidence-based medicine.\n\nCORE BEHAVIOUR:\n- Always retrieve evidence before answering. Never answer from memory alone.\n- Cite the specific source, trial name, or document for every clinical claim.\n- If the retrieved evidence is insufficient, say so explicitly.\n- Be precise with numbers — dosages, p-values, endpoints, sample sizes matter.\n\nCLARIFICATION RULE — MANDATORY:\nWhen the request is ambiguous, you MUST call the ask_user_input tool.\nDo NOT ask clarifying questions in plain text — ALWAYS use the tool.\nFailing to call the tool means the user cannot respond interactively.\nUse ask_user_input when:\n- The trial name or drug is ambiguous\n- The question could refer to multiple phases or indications\n- The user intent is unclear\n\nDISCLAIMERS:\n- Always include: This information is for research purposes only and does not constitute medical advice.\n- Never recommend specific treatments for individual patients.\n- Flag if data is preliminary, unpublished, or from a single study.\n\nTOOL USAGE:\n- Maximum {{max_tool_calls}} tool calls per request.\n- Use search for recent trials and regulatory decisions.\n- Use graph for relationships between drugs, targets, and indications.\n- Use summariser for long documents.\n- Use chart only when visualising data adds clarity.\n\n{{episodic_context}}",
        "inputVariables": [
          {"name": "domain_frame"},
          {"name": "episodic_context"},
          {"name": "max_tool_calls"}
        ]
      }
    }
  }]
}
ENDJSON

aws bedrock-agent create-prompt \
    --cli-input-json file:///tmp/prompt_payload.json \
    --region us-east-1

rm /tmp/prompt_payload.json
```

Note the `id` from the response. Then create version 1:

```bash
aws bedrock-agent create-prompt-version \
    --prompt-identifier "<prompt-id>" \
    --region us-east-1
```

**Updating the prompt — create a new version:**

```bash
cat > /tmp/prompt_payload.json << 'ENDJSON'
{
  "name": "clinical-trial-agent-system-prompt",
  "variants": [{ ... updated text ... }]
}
ENDJSON

aws bedrock-agent update-prompt \
    --prompt-identifier "<prompt-id>" \
    --cli-input-json file:///tmp/prompt_payload.json \
    --region us-east-1

aws bedrock-agent create-prompt-version \
    --prompt-identifier "<prompt-id>" \
    --region us-east-1

# Point SSM to the new version number
aws ssm put-parameter \
    --name /clinical-agent/dev/bedrock/prompt_version \
    --value "<new-version-number>" --type String --overwrite

rm /tmp/prompt_payload.json
```

### 3. Postgres Database

```bash
# Create the database
createdb clinical_agent

# Create Secrets Manager secret using a file to avoid quoting issues
# with special characters in passwords (@, #, % etc.)
cat > /tmp/pg_secret.json << 'ENDJSON'
{
    "host":     "localhost",
    "port":     "5432",
    "dbname":   "clinical_agent",
    "username": "postgres",
    "password": "your-postgres-password"
}
ENDJSON

aws secretsmanager create-secret \
    --name clinical-agent/dev/postgres \
    --secret-string file:///tmp/pg_secret.json

rm /tmp/pg_secret.json
```

LangGraph checkpoint tables are created automatically on first run.

### 4. DynamoDB Trace Table

The DynamoDB table for agent trace persistence is **created automatically** on the
first request — no manual table creation needed. You only need to register the
table name in SSM (step 5 below) so the platform knows what to create.

The table is provisioned with:
- **Billing:** PAY_PER_REQUEST — no capacity planning needed
- **TTL:** `expires_at` attribute — traces auto-deleted after 30 days (free)
- **PK:** `run_id` (String) — unique per agent request

IAM role running the platform must have these DynamoDB permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "dynamodb:CreateTable",
    "dynamodb:DescribeTable",
    "dynamodb:PutItem",
    "dynamodb:GetItem",
    "dynamodb:UpdateTimeToLive"
  ],
  "Resource": "arn:aws:dynamodb:us-east-1:*:table/clinical-agent-traces"
}
```

> **Local dev:** Set `TRACE_TABLE_NAME=clinical-agent-traces` in `.env.local`
> (or omit it — the platform defaults to `clinical-agent-traces` locally).

### 5. SSM Parameters

Generate a platform API key first:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Then set all parameters:

```bash
# Pinecone
aws ssm put-parameter --name /clinical-agent/dev/pinecone/api_key \
    --value "pcsk-..." --type SecureString

aws ssm put-parameter --name /clinical-agent/dev/pinecone/index_name \
    --value "clinical-agent" --type String

# Bedrock prompt (IDs from step 2)
aws ssm put-parameter --name /clinical-agent/dev/bedrock/prompt_id \
    --value "<prompt-id>" --type String

aws ssm put-parameter --name /clinical-agent/dev/bedrock/prompt_version \
    --value "1" --type String

# DynamoDB trace table (table is auto-created on first request)
aws ssm put-parameter --name /clinical-agent/dev/dynamodb/trace_table_name \
    --value "clinical-agent-traces" --type String

# Platform API key
aws ssm put-parameter --name /clinical-agent/dev/platform/api_key \
    --value "<generated-key>" --type SecureString
```

---

## Episodic Memory

The agent remembers past interactions per user across sessions using Pinecone
as the episodic store. This is completely separate from the semantic cache.

### The problem it solves

Without episodic memory, every session starts blank:

```
Session 1:  "What is the metformin dose for my patient with eGFR 25?"
            Agent gives a detailed answer.

Session 2:  "Is that dose still safe?"
            Agent: "Which dose? Which patient?" ← forgot everything
```

With episodic memory, past context is retrieved and injected automatically:

```
Session 2:  Agent sees → "Previous: User asked about metformin dosing
                          for eGFR 25 and received 500mg twice daily"
            Agent: "Based on our previous discussion about eGFR 25 dosing..."
```

### The core idea — relevant retrieval, not full state passing

The naive approach is to pass the entire conversation history to the LLM.
This breaks fast:

```
Session 1:   10 messages  →   ~2,000 tokens
Session 10: 100 messages  →  ~20,000 tokens  ← expensive, hits context limit
```

Episodic memory solves this by storing past Q&A pairs in Pinecone and
retrieving only the **top 3 most relevant** ones for the current question:

```
User asks current question
      ↓
Embed question → search Pinecone (episodic__user_abc namespace)
      ↓
Returns top 3 semantically similar past Q&As
      ↓
@dynamic_prompt injects them into the system prompt
      ↓
LLM sees: current question + 3 relevant memories  (~500 tokens, always bounded)
```

Two benefits of this approach:

**Bounded cost** — 3 sessions or 300 sessions, the LLM always sees the same
number of tokens. The context window never fills up regardless of how long
the user has been interacting with the agent.

**Relevance over recency** — a memory from 20 sessions ago surfaces if it is
semantically related to today's question. Pure state passing would miss it
entirely once older messages fall outside the context window.

This is why it is called *episodic* — borrowed from human psychology.
Episodic memory in humans is how you recall a specific past experience
when something in the present triggers it, not by replaying your entire life history.

### How the LLM decides what to store

The system prompt instructs the LLM to end every response with `EPISODIC: YES`
or `EPISODIC: NO`. The middleware reads this tag and stores or skips accordingly.

A keyword rule cannot make this decision reliably. Both questions contain "eGFR":

```
"What is the normal eGFR range?"         → generic, any user would ask this → NO
"What dose for my patient with eGFR 25?" → specific to this user's patient  → YES
```

Only the LLM understands the difference. And since the LLM is already running,
adding the tag costs zero extra tokens.

The tag is always stripped before the answer reaches the user — it is internal
metadata only.

### How it is different from Semantic Cache

| | Episodic Memory | Semantic Cache |
|---|---|---|
| Scope | Private per user | Shared across all users |
| On retrieval | Agent still runs, context enriched | Agent skipped entirely |
| Stores | User-specific interactions | Generic reusable answers |
| Example | "This user's patient has eGFR 25" | "Metformin Phase 3 results" |
| Purpose | Personalisation | Cost saving |

---

## Semantic Cache

The platform includes a Pinecone-backed semantic cache (`SemanticCacheMiddleware`)
that short-circuits the entire agent pipeline on a cache hit — no tools, no LLM call,
no guardrail re-evaluation. A HIT saves ~2–4s latency and ~$0.01 per request.

### How it works

```
User question → embed → query Pinecone (cache_pharma namespace)
    HIT  (score ≥ threshold) → return cached answer immediately
    MISS                     → run agent → store answer in background
```

### Cache eligibility — what gets cached and why

Not every answer is worth caching. The cache applies an intelligent policy on the
write path (`after_agent`) to ensure only high-quality, reusable answers are stored.

**An answer is cached only if ALL of the following are true:**

| Signal | Threshold | Rationale |
|---|---|---|
| `tool_count > 0` | at least 1 tool called | LLM must have retrieved evidence. Memory-only answers violate the CORE BEHAVIOUR rule and must never be cached in a clinical domain. |
| `faithfulness ≥ 0.85` | OutputGuardrail score | Answers scoring below threshold are partially hallucinated. Caching a hallucination serves it at full speed to every future user. |
| `is_fallback == False` | not a guardrail rejection | OutputGuardrail hard-block responses are replaced with a safe fallback. Caching the fallback means the agent never retries. |
| `len(answer) ≥ 100` | characters | Short responses are typically clarification questions (`ask_user_input` answers) or error messages, not clinical answers. |
| Single-turn question | `len(human_messages) == 1` | Multi-turn answers reference prior context ("as I mentioned about the dosing…"). Serving them cold to a new user produces a confusing, context-dependent response. |
| No patient-specific signals | no "my", "patient", "years old" etc. | Patient-specific answers must not be reused across users — both for accuracy and HIPAA reasons. |

**Dynamic TTL — not all clinical data ages at the same rate:**

| Question type | TTL | Examples |
|---|---|---|
| Regulatory / guidelines | 7 days | FDA approval, ADA guidelines, ICH guidance |
| Clinical trial results | 1 day | Phase 3 RCT endpoints, hazard ratios |
| Safety / market | 1 hour | Recalls, black box warnings, drug availability |
| Default | 3 hours | Everything else |

### Why not use an LLM to decide whether to cache?

The signals above are deterministic and computed for free by existing middleware
(OutputGuardrail already scores faithfulness, ActionGuardrail already counts tool calls).
Adding an LLM caching-decision call would cost ~$0.001 and add ~500ms after every
single response — more than the cache saves on a MISS. Rule-based signals cover ~95%
of cases correctly. The remaining 5% edge cases don't justify an LLM call on the hot path.

An LLM-based cache quality review is appropriate **offline** — a nightly batch job
that evaluates stored entries and evicts low-quality ones — not inline.

### Pinecone namespace strategy

```
cache_pharma   → pharma agent (threshold=0.97, strict)
cache_general  → general agent (threshold=0.88, relaxed)
episodic__*    → episodic memory (separate from cache)
```

Namespaces are isolated so a pharma cached answer can never bleed into a general
agent query, and vice versa.

### Middleware execution order — simple mental model

The stack list order is the **request order**. The response order is simply **reversed**.

```
Request  →   Tracer → PII → ContentFilter → SemanticCache → ... → OutputGuardrail → Agent
Response ←   Tracer ← PII ← ContentFilter ← SemanticCache ← ... ← OutputGuardrail ← Agent
```

Think of it like airport security — you pass through the same checkpoints going in
and coming out, just in reverse.

Your actual logs confirm this:
```
# Request (top to bottom)
[CONTENT_FILTER]  Passed
[CACHE_MW]        MISS — proceeding to agent
...agent runs...

# Response (bottom to top — reverse)
[OUTPUT_GUARD]    PASSED        ← runs first on response
[ACTION_GUARD]    tool_calls=2  ← runs second
[CACHE_MW]        storing...    ← runs third  ✅ guardrail scores already available
[TRACER]          run_complete  ← runs last
```

**The one thing students need to remember:**

> On the response side, whatever is at the **bottom of the stack runs first**.

This is why `SemanticCacheMiddlewareWithRules` works correctly at position 4.
By the time it runs on the response side, `OutputGuardrail` (position 9) and
`ActionGuardrail` (position 8) have already finished and written their scores
into state. The cache can read `faithfulness` and `tool_count` and make the right decision.

If SemanticCache were moved to position 9 (bottom), it would run first on the
response side — before the guardrails — and those scores would not exist yet.

### Future: pub/sub write path

Currently the cache write runs in a **daemon thread** (fire-and-forget) — the
response is returned to the user before the Pinecone upsert completes.

Planned Phase 2: replace the daemon thread with **SNS → SQS → Lambda**. The agent
publishes a cache-write event; a separate Lambda consumer calls `cache.store()`.
Benefits: decoupled, retryable, dead-letter queue for failed writes, cache writer
scales independently of the agent.

---

## Run

```bash
# Clinical trial agent (direct)
APP_ENV=dev python clinical_trial_agent/run.py

# Platform API
APP_ENV=dev uvicorn vs_platform.main:app --host 0.0.0.0 --port 8000 --reload
```

**Call the API:**
```bash
curl -X POST http://localhost:8000/api/v1/clinical-trial/chat \
  -H "X-API-Key: <your-platform-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are Phase 3 results for metformin?", "thread_id": "t1", "domain": "pharma"}'
```

---

## Known Issues & Fixes

**`@` in Postgres password breaks DSN parsing**
Passwords with special characters (`@`, `#`, `%`) are URL-encoded automatically
by `core/aws.py`. No action needed — just ensure the raw password is correct in Secrets Manager.

**`PostgresSaver` requires `autocommit=True`**
`CREATE INDEX CONCURRENTLY` cannot run inside a transaction.
`agent.py` connects with `psycopg.connect(conn_string, autocommit=True)` — already handled.

**HITL: agent responds in text instead of calling `ask_user_input`**
The Bedrock prompt must include the MANDATORY CLARIFICATION RULE (already in the prompt
template above). If you have an older prompt without it, update using the file approach
in step 2 and bump the SSM version.

**Shell quoting errors (`dquote>` prompt) when passing JSON to AWS CLI**
Always write JSON to a file and pass `file:///tmp/filename.json` to `--cli-input-json`
or `--secret-string`. Never pass complex JSON inline on the command line.

**DynamoDB traces not appearing after first request**
DynamoDB TTL deletes are eventually consistent — items are removed within 48 h of
`expires_at`, not instantly. If traces are missing immediately after the first request,
check CloudWatch logs for `[AWS] Trace persisted` or `[TRACER] Background DynamoDB write failed`.
The background write is fire-and-forget; errors are logged but never bubble up to the API response.

**Cache HIT on every request — `_store_sync` never fires**
This is expected when the cache is already warm from previous runs. `_store_sync` only
fires on a MISS where the agent runs and produces a cacheable answer. To force a cold
cache for testing, clear the Pinecone namespace:
```python
from pinecone import Pinecone
pc = Pinecone(api_key="...")
pc.Index("clinical-agent").delete(delete_all=True, namespace="cache_pharma")
```

**Cache storing ambiguous clarification answers**
If `ask_user_input` was called but no HITL resume happened (e.g. in run.py),
the short clarification text can appear as the "answer". The `len(answer) >= 100`
guard filters these out — they are never written to the cache.

---

## PyCharm Setup

- Mark `clinical_trial_agent/` and `vs-agent-core/` as **Sources Root**
- Run config working directory: `.../vs-agentic-platform` (not `clinical_trial_agent/`)
- Uncheck **Add content/source roots to PYTHONPATH** in run config

> `vs_platform/` is named with underscore to avoid shadowing Python's built-in
> `platform` standard library module.

---

## Dependency Notes

These packages are pinned and must move together:

```
langsmith==0.6.9
langchain-core==1.2.22
langchain==1.1.0
langchain-openai==1.1.1
langgraph==1.0.4
psycopg==3.3.3          ← 3.2.x breaks on Python 3.13
psycopg-binary==3.3.3
```