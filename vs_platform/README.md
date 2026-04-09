# VS Platform

FastAPI gateway for the VS Agentic Platform. Handles authentication, rate limiting,
prompt injection detection, agent routing, HITL, and prompt version management.

---

## Structure

```
vs_platform/
  main.py                        ← FastAPI app, middleware, router registration
  gateway/
    router.py                    ← POST /chat, POST /resume, GET /health
    schemas.py                   ← All Pydantic request/response models
    auth.py                      ← API key + Cognito JWT authentication
    injection.py                 ← Gateway-level prompt injection guard
    rate_limiter.py              ← Per-user sliding window rate limiter
    middleware.py                ← RequestContextMiddleware, TimingMiddleware
  observability/
    logger.py                    ← Structured JSON logger (CloudWatch ready)
    tracer.py                    ← Request ID ContextVar propagation
  prompt_versioning/
    manager.py                   ← Bedrock prompt version CRUD via SSM
    router.py                    ← GET/POST /prompts/{agent}/{env}/...
  tests/
    test_auth.py
    test_prompt_versioning.py
```

---

## Start the server

```bash
# Development
APP_ENV=dev uvicorn vs_platform.main:app --host 0.0.0.0 --port 8000 --reload

# Production (ECS)
uvicorn vs_platform.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Swagger UI is available at `http://localhost:8000/docs` when `APP_ENV != prod`.

---

## Environment variables

| Variable        | Required | Default | Description |
|---|---|---|---|
| `APP_ENV`       | No  | `prod`  | `prod` \| `staging` \| `dev` — controls docs visibility |
| `LOG_LEVEL`     | No  | `INFO`  | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `OPENAI_API_KEY`| Yes | —       | Required for LLM calls |
| `PLATFORM_API_KEY` | Local only | `local-dev-key` | API key for local dev (no SSM needed) |

AWS credentials come from the ECS task role / EC2 instance profile — no credential env vars needed in production.

---

## API Reference

### Agent endpoints

```
POST /api/v1/{agent}/chat
POST /api/v1/{agent}/resume
GET  /api/v1/health
```

**Chat request:**
```json
{
  "message":   "What are Phase 3 results for metformin?",
  "thread_id": "thread-abc123",
  "domain":    "pharma"
}
```

**Normal response:**
```json
{
  "answer":      "Metformin achieved a 42% reduction in HbA1c...",
  "thread_id":   "thread-abc123",
  "request_id":  "4f9a12bc3d",
  "interrupted": false,
  "agent":       "clinical-trial",
  "latency_ms":  1243.5
}
```

**HITL interrupt response** (`interrupted=true`):
```json
{
  "answer":      "",
  "interrupted": true,
  "interrupt_payload": {
    "question":       "Which drug or condition are you asking about?",
    "options":        ["Metformin", "Semaglutide", "Empagliflozin"],
    "allow_freetext": true
  },
  "thread_id":   "thread-abc123",
  "request_id":  "4f9a12bc3d",
  "agent":       "clinical-trial",
  "latency_ms":  843.2
}
```

**Resume after HITL:**
```json
POST /api/v1/clinical-trial/resume
{
  "thread_id":   "thread-abc123",
  "user_answer": "Metformin",
  "domain":      "pharma"
}
```

### Prompt versioning endpoints (admin only)

```
GET  /api/v1/prompts/{agent}/{env}           ← list all versions
GET  /api/v1/prompts/{agent}/{env}/active    ← get active version
POST /api/v1/prompts/{agent}/{env}/activate  ← activate a version
POST /api/v1/prompts/{agent}/{env}/rollback  ← rollback to previous
```

---

## Authentication

Two modes, selected by header:

**API key (always available):**
```bash
curl -H "X-API-Key: <your-key>" ...
```
Local dev: set `PLATFORM_API_KEY=any-value` in `.env.local` and use that as the key.

**Cognito JWT (when configured):**
```bash
curl -H "Authorization: Bearer <jwt>" ...
```
Only active when `COGNITO_USER_POOL_ID` SSM parameter exists.

---

## HITL flow

```
1. POST /chat   → interrupted=True + interrupt_payload (question + options)
2. User reads the question and selects an answer
3. POST /resume → same thread_id + user_answer + domain
4. Agent continues → returns final answer
```

The agent state is preserved between steps 1 and 3 via PostgresSaver using `thread_id`.

---

## Rate limits

| Auth mode | Limit |
|---|---|
| API key (service account) | 10,000 req / 60s |
| Premium (Cognito scope)   | 300 req / 60s |
| Default                   | 60 req / 60s |

Rate limit exceeded returns HTTP 429 with `Retry-After` header.

---

## Prompt versioning

Prompt templates live in Bedrock Prompt Management. The active version pointer lives in SSM:

```
/clinical-trial-agent/{env}/bedrock/prompt_version
```

Activating a new version updates this SSM parameter — takes effect on the next request with **zero downtime**. No restart required.

**Activate a version:**
```bash
curl -X POST http://localhost:8000/api/v1/prompts/clinical-trial/dev/activate \
  -H "X-API-Key: <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"version": "3", "reason": "improved HITL instructions"}'
```

**Rollback:**
```bash
curl -X POST http://localhost:8000/api/v1/prompts/clinical-trial/dev/rollback \
  -H "X-API-Key: <admin-key>"
```

Only one level of rollback is supported. For deeper history, use `/activate` with an explicit version number.

---

## Observability

Every log line is structured JSON — ready for CloudWatch Logs Insights or Datadog:

```json
{
  "ts":         "2026-04-09T10:35:42.123Z",
  "level":      "INFO",
  "logger":     "vs_platform.gateway.router",
  "msg":        "Request complete",
  "request_id": "4f9a12bc3d",
  "agent":      "clinical-trial",
  "method":     "POST",
  "path":       "/api/v1/clinical-trial/chat",
  "status":     200,
  "latency_ms": 1243.5
}
```

`request_id` is automatically injected into every log line from a ContextVar — no need to pass it manually in `extra={}`.

---

## Bugs fixed

| File | Issue | Fix |
|---|---|---|
| `main.py` | Uvicorn command said `platform.main:app` | Changed to `vs_platform.main:app` |
| `schemas.py` | `HITLResumeRequest` had no `domain` field | Added `domain` with default `"pharma"` |
| `gateway/router.py` | Passed full history to agent — duplicated messages with PostgresSaver | Now passes only the current message |
| `gateway/router.py` | `resume()` had no `context` — user_id was "anonymous" | Added context with auth.user_id and body.domain |
| `gateway/injection.py` | `get_env` imported but never used | Removed dead import |
| `gateway/rate_limiter.py` | `check_rate_limit(auth: AuthContext)` — FastAPI cannot inject AuthContext without Depends | Changed to `Depends(require_auth)` |
| `observability/logger.py` | `request_id` ContextVar not injected into log records automatically | Added `_RequestContextFilter` that reads ContextVar on every log record |
| `prompt_versioning/manager.py` | `_validate_version_exists` passed `app_name` to `list_versions()` which expects `agent_slug` | Fixed to pass `agent_slug` |
| `prompt_versioning/manager.py` | `_put_ssm` called `aws._ssm()` — private method | Changed to `boto3.client("ssm")` directly |

---

## Adding a new agent

1. Implement `build_agent(domain)` in your agent package.
2. Add an entry to `AGENT_REGISTRY` in `gateway/router.py`:
   ```python
   AGENT_REGISTRY = {
       "clinical-trial": _load_clinical_trial_agent,
       "my-new-agent":   _load_my_new_agent,      # ← add this
   }
   ```
3. Add the agent slug to `AGENT_APP_NAMES` in `prompt_versioning/manager.py` if it uses Bedrock prompts.

No other code changes needed.
