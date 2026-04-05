# VS Agentic Platform

**Vidya Sankalp** — Multi-agent AI platform for Applied GenAI Engineering.

---

## Repository Structure

```
vs-agentic-platform/
  vs-agent-core/              ← pip installable base package
  clinical_trial_agent/       ← pharma clinical trial domain agent
  platform/                   ← FastAPI gateway, auth, rate limiting, prompt versioning
  requirements-dev.txt        ← editable install for all three packages
  README.md
```

---

## Packages

| Package | Name | Purpose |
|---|---|---|
| `vs-agent-core` | `vs-agent-core` | Domain-agnostic foundation — AWS helpers, Pinecone store, semantic cache, base middleware |
| `clinical_trial_agent` | `clinical-trial-agent` | Pharma domain agent — HITL, guardrails, tools, HIPAA PII, Bedrock prompt |
| `platform` | `vs-agent-platform` | FastAPI gateway — auth, rate limiting, injection check, prompt versioning API |

---

## Local Development Setup

```bash
# 1. Clone
git clone https://github.com/vidyasankalp/vs-agentic-platform
cd vs-agentic-platform

# 2. Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install all packages in editable mode
pip install -r requirements-dev.txt

# 4. Set environment variables
export OPENAI_API_KEY="sk-..."
export APP_ENV="dev"            # uses /clinical-trial-agent/dev/* SSM paths

# 5. Start the platform
uvicorn platform.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## AWS Setup (one-time per environment)

### SSM Parameter Store

```bash
# Pinecone
aws ssm put-parameter --name /clinical-trial-agent/prod/pinecone/api_key \
    --value "pcsk-..." --type SecureString
aws ssm put-parameter --name /clinical-trial-agent/prod/pinecone/index_name \
    --value "clinical-agent" --type String

# Bedrock Prompt Management
aws ssm put-parameter --name /clinical-trial-agent/prod/bedrock/prompt_id \
    --value "<bedrock-prompt-id>" --type String
aws ssm put-parameter --name /clinical-trial-agent/prod/bedrock/prompt_version \
    --value "1" --type String

# Platform API key (for internal service-to-service calls)
aws ssm put-parameter --name /clinical-trial-agent/prod/platform/api_key \
    --value "your-internal-api-key" --type SecureString

# Cognito (for external JWT auth)
aws ssm put-parameter --name /clinical-trial-agent/prod/cognito/user_pool_id \
    --value "us-east-1_XXXXXXX" --type String
aws ssm put-parameter --name /clinical-trial-agent/prod/cognito/region \
    --value "us-east-1" --type String
```

### Secrets Manager (Postgres)

```bash
aws secretsmanager create-secret \
    --name clinical-agent/prod/postgres \
    --secret-string '{
        "host":     "your-rds-endpoint.rds.amazonaws.com",
        "port":     "5432",
        "dbname":   "clinical_agent",
        "username": "agent_user",
        "password": "..."
    }'
```

### Pinecone Index

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="<your-key>")
pc.create_index(
    "clinical-agent",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

---

## API Reference

### Chat

```bash
# Send a message
curl -X POST http://localhost:8000/api/v1/clinical-trial/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message":   "What are the Phase 3 results for metformin?",
    "thread_id": "thread-abc-123",
    "domain":    "pharma"
  }'

# Resume a HITL pause
curl -X POST http://localhost:8000/api/v1/clinical-trial/resume \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id":   "thread-abc-123",
    "user_answer": "Efficacy in HbA1c reduction"
  }'
```

### Prompt Versioning

```bash
# List all versions
curl http://localhost:8000/api/v1/prompts/clinical-trial/prod \
  -H "X-API-Key: your-api-key"

# Activate version 3
curl -X POST http://localhost:8000/api/v1/prompts/clinical-trial/prod/activate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"version": "3", "reason": "Improved clinical disclaimer wording"}'

# Rollback to previous
curl -X POST http://localhost:8000/api/v1/prompts/clinical-trial/prod/rollback \
  -H "X-API-Key: your-api-key"
```

---

## Running Tests

```bash
# Fast unit tests (no LLM, no AWS)
pytest platform/tests/ -v

# Clinical trial agent tests
pytest clinical_trial_agent/tests/ -v

# All tests excluding LLM calls
pytest -m "not llm" -v

# Generate HTML report
pytest --html=reports/test_report.html
```

---

## Request Flow

```
Client
  │
  ├─ POST /api/v1/clinical-trial/chat
  │
  ▼
FastAPI (platform/main.py)
  │
  ├─ RequestContextMiddleware  → assign request_id
  ├─ TimingMiddleware          → start timer
  │
  ▼
gateway/router.py
  │
  ├─ require_auth              → validate JWT or API key
  ├─ check_rate_limit          → sliding window per user
  ├─ check_injection           → prompt injection guard
  │
  ▼
clinical_trial_agent
  │
  ├─ TracerMiddleware           core
  ├─ DomainPIIMiddleware        pharma — HIPAA PII scrubbing
  ├─ ContentFilterMiddleware    pharma — toxic pattern check
  ├─ SemanticCacheMiddleware    core   — Pinecone cache lookup
  ├─ EpisodicMemoryMiddleware   core   — past context enrichment
  ├─ SummarizationMiddleware    core
  ├─ HumanInTheLoopMiddleware   core   — ask_user_input gate
  ├─ ActionGuardrailMiddleware  pharma — tool call cap
  └─ OutputGuardrailMiddleware  pharma — faithfulness + safety judge
  │
  ▼
Response → client with X-Request-ID + X-Latency-Ms headers
```

---

## Adding a New Agent

1. Build your domain agent following the `clinical_trial_agent/` pattern.
2. Register it in `platform/gateway/router.py`:
   ```python
   AGENT_REGISTRY["finance"] = _load_finance_agent
   ```
3. Add SSM paths for the new agent's Bedrock prompt and Cognito config.
4. Done — auth, rate limiting, injection check, and prompt versioning
   work automatically for all registered agents.
