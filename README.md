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

### 4. SSM Parameters

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

# Platform API key
aws ssm put-parameter --name /clinical-agent/dev/platform/api_key \
    --value "<generated-key>" --type SecureString
```

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