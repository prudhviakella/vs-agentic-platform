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
pip install neo4j  # graph_tool dependency

# 3. Export environment variables
export OPENAI_API_KEY="sk-..."
export PINECONE_API_KEY="pcsk-..."
export PINECONE_INDEX_NAME="clinical-agent"
export CLINICAL_TRIALS_INDEX="clinical-trials-index"
export NEO4J_URI="neo4j+s://52c31090.databases.neo4j.io"
export NEO4J_USER="52c31090"
export NEO4J_PASSWORD="your-neo4j-password"
export TAVILY_API_KEY="tvly-..."
export POSTGRES_URL="postgresql://user:pass@localhost:5432/clinical_agent"
export PLATFORM_API_KEY="local-dev-key"
```

---

## AWS Prerequisites

Complete these steps in order before running the agent.

### 1. Pinecone Indexes

The platform uses **two Pinecone indexes**:

| Index | Purpose | Namespace |
|---|---|---|
| `clinical-agent` | Semantic cache + episodic memory | `cache_pharma`, `episodic__*` |
| `clinical-trials-index` | Clinical trial knowledge base (real data) | `clinical-trials` |

Create via Python:

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="your-pinecone-api-key")

# Index 1 — cache + episodic memory
pc.create_index(
    name="clinical-agent",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Index 2 — clinical trial knowledge base
pc.create_index(
    name="clinical-trials-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

Wait ~30 seconds for indexes to initialize before running.

### 2. Bedrock System Prompt

The system prompt lives in AWS Bedrock Prompt Management — not in code.
This allows clinical writers to update prompts without touching the codebase.

**Create the prompt:**

```bash
cat > /tmp/prompt_payload.json << 'ENDJSON'
{
  "name": "clinical-trial-agent-system-prompt-1",
  "description": "System prompt for the clinical trial agent",
  "variants": [
    {
      "name": "default",
      "templateType": "TEXT",
      "templateConfiguration": {
        "text": {
          "text": "{{domain_frame}}\n\nYou are an expert clinical research assistant with deep knowledge of pharmaceutical drug development, clinical trial design, regulatory frameworks (FDA, EMA, ICH), and evidence-based medicine.\n\nCORE BEHAVIOUR:\n- Always retrieve evidence before answering. Never answer from memory alone.\n- Cite the specific source, trial name, or document for every clinical claim.\n- If the retrieved evidence is insufficient, say so explicitly.\n- Be precise with numbers — dosages, p-values, endpoints, sample sizes matter.\n\nCLARIFICATION RULE — MANDATORY:\nAsking questions in plain text is FORBIDDEN. It breaks the interactive UI.\nThe ask_user_input tool is the ONLY permitted way to ask the user a question.\n\nStep 1: If the request needs any clarification, call ask_user_input ONCE.\nStep 2: Once the user answers (any answer), call search_tool IMMEDIATELY.\nStep 3: Answer based on the search results.\n\nSTRICT LIMITS:\n- NEVER write a question in your response text — use ask_user_input instead.\n- NEVER call ask_user_input more than once per conversation.\n- After receiving ANY user answer, go straight to search_tool.\n- The user's answer is always sufficient — search with whatever they gave you.\n\nDISCLAIMERS:\n- Always include: This information is for research purposes only and does not constitute medical advice.\n- Never recommend specific treatments for individual patients.\n- Flag if data is preliminary, unpublished, or from a single study.\n\nTOOL USAGE:\n- Maximum {{max_tool_calls}} tool calls per request.\n- Use search_tool for clinical trial data and drug evidence.\n- Use graph_tool for relationships between drugs, targets, and indications.\n- Use summariser_tool for long documents.\n- Use chart_tool only when visualising data adds clarity.\n\nEPISODIC TAGGING — MANDATORY:\nEnd every response with exactly one of these tags on its own line:\nEPISODIC: YES — if your answer is specific to this user's case, patient, or context\nEPISODIC: NO  — if your answer is generic knowledge that anyone would ask\n\nExamples:\n- 'What is the metformin dose for eGFR 25?' → EPISODIC: YES (patient-specific)\n- 'What is the normal eGFR range?' → EPISODIC: NO (generic knowledge)\n- 'What are the Phase 3 results for Pfizer BNT162b2?' → EPISODIC: NO (public data)",
          "inputVariables": [
            {"name": "domain_frame"},
            {"name": "max_tool_calls"}
          ]
        }
      }
    }
  ]
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
# Edit the JSON file, then:
aws bedrock-agent update-prompt \
    --prompt-identifier "<prompt-id>" \
    --cli-input-json file:///tmp/prompt_payload.json \
    --region us-east-1

aws bedrock-agent create-prompt-version \
    --prompt-identifier "<prompt-id>" \
    --region us-east-1

# Point SSM to the new version
aws ssm put-parameter \
    --name /clinical-trial-agent/dev/bedrock/prompt_version \
    --value "<new-version-number>" --type String --overwrite

rm /tmp/prompt_payload.json
```

### 3. Postgres Database

```bash
# Create the database
createdb clinical_agent

# Create Secrets Manager secret
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

LangGraph checkpoint tables (HITL state) are created automatically on first run.

### 4. DynamoDB Trace Table

Created automatically on first request. IAM role must have:

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

### 5. SSM Parameters

```bash
# Generate a platform API key
python -c "import secrets; print(secrets.token_hex(32))"

# Pinecone (semantic cache + episodic memory index)
aws ssm put-parameter --name /clinical-agent/dev/pinecone/api_key \
    --value "pcsk-..." --type SecureString

aws ssm put-parameter --name /clinical-agent/dev/pinecone/index_name \
    --value "clinical-agent" --type String

# Bedrock prompt (IDs from step 2)
aws ssm put-parameter --name /clinical-trial-agent/dev/bedrock/prompt_id \
    --value "<prompt-id>" --type String

aws ssm put-parameter --name /clinical-trial-agent/dev/bedrock/prompt_version \
    --value "1" --type String

# DynamoDB
aws ssm put-parameter --name /clinical-agent/dev/dynamodb/trace_table_name \
    --value "clinical-agent-traces" --type String

# Platform API key
aws ssm put-parameter --name /clinical-agent/dev/platform/api_key \
    --value "<generated-key>" --type SecureString
```

---

## Architecture

### Middleware Stack (9 layers)

The middleware stack wraps the agent. Layers run top→bottom on request, bottom→top on response.

```
Request  →  [1]Tracer → [2]PII → [3]ContentFilter → [4]SemanticCache
          → [5]EpisodicMemory → [6]Summarization → [7]HITL
          → [8]ActionGuardrail → [9]OutputGuardrail → Agent

Response ←  [1]Tracer ← [2]PII ← [3]ContentFilter ← [4]SemanticCache
          ← [5]EpisodicMemory ← [6]Summarization ← [7]HITL
          ← [8]ActionGuardrail ← [9]OutputGuardrail ← Agent
```

Think of it like airport security — same checkpoints in and out, just reversed.

| # | Layer | Hook | Purpose |
|---|---|---|---|
| 1 | `TracerMiddleware` | before+after | DynamoDB trace persistence, latency logging |
| 2 | `DomainPIIMiddleware` | before+after | Email/CC redaction on input and output |
| 3 | `ContentFilterMiddleware` | before | Block toxic/off-domain content |
| 4 | `SemanticCacheMiddleware` | before+after | Cache HIT skips agent entirely |
| 5 | `EpisodicMemoryMiddleware` | before+after | Inject past context, store new Q&A |
| 6 | `SummarizationMiddleware` | wrap_model | Compress state when >3000 tokens |
| 7 | `HumanInTheLoopMiddleware` | wrap_model | Pause on `ask_user_input`, resume on answer |
| 8 | `ActionGuardrailMiddleware` | after | Enforce tool call limits |
| 9 | `OutputGuardrailMiddleware` | after | 3-layer faithfulness check |

### Tools

| Tool | Purpose | HITL? |
|---|---|---|
| `ask_user_input` | Pause for human clarification | YES — only tool that pauses |
| `search_tool` | Vector search on `clinical-trials-index` (Pinecone) | No |
| `graph_tool` | Cypher queries on Neo4j clinical trials graph | No |
| `summariser_tool` | Synthesise multiple retrieved chunks | No |
| `chart_tool` | Generate chart spec from data points | No |

### System Prompt

The system prompt is fetched from **AWS Bedrock Prompt Management** on agent startup.
`aws.py` reads `prompt_id` and `prompt_version` from SSM, fetches the template from Bedrock,
and substitutes two placeholders:

- `{{domain_frame}}` — pharma vs general framing (set at agent creation time)
- `{{max_tool_calls}}` — integer cap from `tools/__init__.py`

There is no local fallback — all environments run against real AWS.

---

## HITL — Human in the Loop

The agent pauses mid-conversation when it needs clarification and resumes after the human answers.

### Flow

```
POST /chat  {"message": "Show me the trial data", "thread_id": "t1"}
    ↓
LLM calls ask_user_input("Which trial?", ["Pfizer BNT162b2", "Remdesivir", ...])
    ↓
HumanInTheLoopMiddleware intercepts → graph pauses → checkpoint saved to Postgres
    ↓
Response: {"interrupted": true, "interrupt_payload": {"question": ..., "options": [...]}}

POST /resume {"thread_id": "t1", "user_answer": "Pfizer BNT162b2 Phase 3"}
    ↓
Agent resumes from Postgres checkpoint → LLM calls search_tool → answers
    ↓
Response: {"interrupted": false, "answer": "..."}
```

### Rules (enforced by system prompt)

- `ask_user_input` is the **only** tool that pauses the agent
- The LLM may call it **at most once** per conversation
- After receiving any answer, it **must immediately** call `search_tool`
- All other tools auto-execute without pausing

### Why `system_prompt=` not `@dynamic_prompt`

`@dynamic_prompt` uses `wrap_model_call` — the same hook type as `HumanInTheLoopMiddleware`.
Having both in the middleware chain caused HITL to never fire (LLM could call `ask_user_input`
6 times in one turn without pausing). Passing `system_prompt=` directly to `create_agent()`
removes the conflict entirely. This is the correct LangChain 1.0 API.

### Resume payload format

```json
{
  "thread_id":   "your-thread-id",
  "user_answer": "Pfizer BNT162b2 COVID-19 vaccine Phase 3",
  "domain":      "pharma"
}
```

---

## Episodic Memory

The agent remembers past interactions per user across sessions.

### The core idea — relevant retrieval, not full history

Storing full conversation history scales badly:

```
Session 1:    10 messages  →   ~2,000 tokens
Session 10:  100 messages  →  ~20,000 tokens  ← expensive, hits context limit
```

Episodic memory stores Q&A pairs in Pinecone and retrieves only the **top 3 most relevant**
ones for the current question — cost is bounded regardless of history length.

### How the LLM decides what to store

The system prompt instructs the LLM to end every response with `EPISODIC: YES` or `EPISODIC: NO`.
`EpisodicMemoryMiddleware` reads this tag and stores or skips accordingly.

```
"What is the normal eGFR range?"          → EPISODIC: NO  (generic)
"Dose for my patient with eGFR 25?"       → EPISODIC: YES (patient-specific)
```

The tag is always stripped before the answer reaches the user — internal metadata only.

### How episodic memory and summarization work together

`SummarizationMiddleware` compresses old state messages when they exceed 3,000 tokens.
The actual memories in Pinecone are unaffected — the next turn fetches fresh relevant
memories from Pinecone and injects them again.

```
State (in memory)    → always small, summarized   (working memory)
Pinecone (episodic)  → permanent, full fidelity   (long-term memory)
```

### Episodic Memory vs Semantic Cache

| | Episodic Memory | Semantic Cache |
|---|---|---|
| Scope | Private per user | Shared across all users |
| On retrieval | Agent still runs, context enriched | Agent skipped entirely |
| Stores | User-specific interactions | Generic reusable answers |
| Purpose | Personalisation | Cost saving |

---

## Semantic Cache

`SemanticCacheMiddleware` short-circuits the entire pipeline on a cache hit — no tools,
no LLM call. A HIT saves ~2–4s latency and ~$0.01 per request.

```
User question → embed → query Pinecone (cache_pharma namespace)
    HIT  (score ≥ 0.97) → return cached answer immediately
    MISS                 → run agent → store answer in background
```

### Cache eligibility

An answer is cached only if ALL of these are true:

| Signal | Rationale |
|---|---|
| Single-turn question | Multi-turn answers reference context that won't exist for another user |
| No patient-specific signals | HIPAA — patient answers must never be reused across users |
| `is_fallback == False` | Never cache a guardrail rejection |
| `len(answer) ≥ 100` | Filters out clarification messages |

### Pinecone namespace strategy

```
clinical-agent index:
  cache_pharma   → pharma agent answers  (threshold=0.97)
  cache_general  → general agent answers (threshold=0.88)
  episodic__*    → per-user episodic memory (separate from cache)

clinical-trials-index:
  clinical-trials → ingested clinical trial chunks (search_tool reads here)
```

---

## OutputGuardrail — 3-Layer Safety Check

Runs on every agent response before it reaches the user. Cheapest check first.

```
Layer 1 — regex (<1ms)
  Catches obvious violations: dosage claims, direct treatment recommendations

Layer 2 — faithfulness (gpt-4o-mini)
  "Is this answer grounded in what was actually retrieved?"
  Score 0.0–1.0. Below 0.85 → safe fallback.

Layer 3 — contradiction (gpt-4o-mini)
  "Does this answer contradict the retrieved context?"
  Score 0.0–1.0. Below 0.50 → hard fail.
```

Only runs layers 2 and 3 when real tool results exist (`search_tool`, `graph_tool`).
`ask_user_input` answers are excluded from grounding — user answers like "Pfizer BNT162b2"
are not clinical evidence and would score 0.00.

A sentinel `if state.get("_cache_is_fallback"): return None` at the top of `after_agent`
prevents the middleware from re-evaluating its own fallback messages (which would cause
an infinite retry loop).

---

## Real Data Sources

### search_tool → Pinecone `clinical-trials-index`

Queries real clinical trial chunks ingested from source documents. Uses `text-embedding-3-small`
embeddings (1536-dim, same model used at ingestion). Returns `text`, `breadcrumbs`, and
similarity `score` per chunk.

### graph_tool → Neo4j AuraDB

Queries a live graph of 25 clinical trials ingested from ClinicalTrials.gov via
`clinical_trials_loader.py`. The graph schema:

```
Trial ──TARGETS──────► Disease
      ──USES──────────► Drug
      ──SPONSORED_BY──► Sponsor
      ──MANAGED_BY────► CRO
      ──CONDUCTED_IN──► Country
      ──LOCATED_AT────► Site ──IN_COUNTRY──► Country
      ──MEASURES──────► Outcome
      ──INCLUDES──────► PatientPopulation
      ──ASSOCIATED_WITH► MeSHTerm
      ──BELONGS_TO────► TrialCategory
```

Two-stage query: first searches by trial title/NCT ID, then falls back to searching by
drug name, disease name, sponsor, or MeSH term.

---

## Run

```bash
# Platform API
uvicorn vs_platform.main:app --host 0.0.0.0 --port 8000 --reload
```

### Test the HITL flow

**Step 1 — ambiguous query (should interrupt):**
```bash
curl -X POST http://localhost:8000/api/v1/clinical-trial/chat \
  -H "X-API-Key: local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me the trial data", "thread_id": "test-1", "domain": "pharma"}'
# Expected: interrupted=true with question + options
```

**Step 2 — resume with specific answer:**
```bash
curl -X POST http://localhost:8000/api/v1/clinical-trial/resume \
  -H "X-API-Key: local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test-1", "user_answer": "Pfizer BNT162b2 Phase 3", "domain": "pharma"}'
# Expected: interrupted=false with clinical answer
```

**Step 3 — specific query (no HITL):**
```bash
curl -X POST http://localhost:8000/api/v1/clinical-trial/chat \
  -H "X-API-Key: local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the Phase 3 efficacy results for Pfizer BNT162b2?", "thread_id": "test-2", "domain": "pharma"}'
# Expected: interrupted=false, direct answer, no HITL
```

---

## Known Issues & Fixes

**HITL: LLM calls `ask_user_input` multiple times without pausing**
Caused by `@dynamic_prompt` middleware conflicting with `HumanInTheLoopMiddleware` (both use
`wrap_model_call`). Fixed by removing `@dynamic_prompt` and passing `system_prompt=` directly
to `create_agent()`. The prompt is now fetched from Bedrock at agent creation time.

**OutputGuardrail infinite retry loop (`faithfulness=0.00` repeated)**
Caused by the guardrail re-evaluating its own fallback message. Fixed by adding a sentinel
`if state.get("_cache_is_fallback"): return None` at the top of `after_agent`.

**Episodic memory never storing (`LLM tagged NO` on every response)**
Caused by the `EPISODIC: YES/NO` tag instruction missing from the Bedrock prompt. Fixed by
adding the EPISODIC TAGGING section to the prompt. Without it the LLM never appends the tag
so `_parse_storage_decision()` always returns `False`.

**`[BASE_MW] session_id missing` warning on every request**
Caused by `_get_run_id()` trying `runtime.run_id` and `runtime.thread_id` which don't exist
in LangChain 1.0 (`Runtime` only exposes `context`, `store`, `stream_writer`). Fixed by reading
`runtime.context["session_id"]` which is set by the gateway via `context={"session_id": thread_id}`.

**`@` in Postgres password breaks DSN parsing**
Passwords with special characters are URL-encoded automatically by `core/aws.py`. No action needed.

**`PostgresSaver` requires `autocommit=True`**
`CREATE INDEX CONCURRENTLY` cannot run inside a transaction. `agent.py` connects with
`psycopg.connect(conn_string, autocommit=True)` — already handled.

**Shell quoting errors (`dquote>` prompt) when passing JSON to AWS CLI**
Always write JSON to a file and pass `file:///tmp/filename.json`. Never pass complex JSON inline.

**DynamoDB traces not appearing after first request**
DynamoDB TTL deletes are eventually consistent — items are removed within 48h of `expires_at`,
not instantly. Check logs for `[AWS] Trace persisted` or `[TRACER] Background DynamoDB write failed`.

**`TavilySearchResults` deprecation warning**
`TavilySearchResults` is deprecated in LangChain 0.3.25. Migrate to:
```python
pip install -U langchain-tavily
from langchain_tavily import TavilySearch
```

---

## PyCharm Setup

- Mark `clinical_trial_agent/` and `vs-agent-core/` as **Sources Root**
- Run config working directory: `.../vs-agentic-platform`
- Uncheck **Add content/source roots to PYTHONPATH** in run config

> `vs_platform/` is named with underscore to avoid shadowing Python's built-in `platform` module.

---

## Dependency Notes

These packages are pinned and must move together:

```
langsmith==0.6.9
langchain-core==1.2.22
langchain==1.1.0
langchain-openai==1.1.1
langgraph==1.0.4
langgraph-checkpoint==3.0.1
langgraph-checkpoint-postgres==3.0.5
langgraph-prebuilt==1.0.7
psycopg==3.3.3          ← 3.2.x breaks on Python 3.13
psycopg-binary==3.3.3
neo4j                   ← required for graph_tool
```