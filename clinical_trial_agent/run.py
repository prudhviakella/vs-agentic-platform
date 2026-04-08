"""
run.py — HITL Loop and Usage Examples
=======================================
Provides:
  handle_ask_user_input()  — console handler for ask_user_input interrupts
  invoke_with_hitl()       — agent invocation loop with pause/resume handling
  run_examples()           — five demo invocations covering all paths

How the HITL loop works:
  1. Call agent.invoke() with messages
  2. If response has __interrupt__: LLM called ask_user_input — needs user input
       → show question + options in console
       → get human answer
       → resume with Command(resume={"decisions": [{type: edit, edited_action: ...}]})
  3. If no __interrupt__: agent finished → return final response

The interrupt payload structure (identical to notebook 3.3):
  response["__interrupt__"][0].value = {
      "action_requests": [{
          "name": "ask_user_input",
          "args": {
              "question":       "Which city in India are you departing from?",
              "options":        ["Delhi (DEL)", "Mumbai (BOM)", "Hyderabad (HYD)", ...],
              "allow_freetext": True,
              "user_answer":    ""   ← we inject the answer here via "edit"
          }
      }],
      "review_configs": [{"action_name": "ask_user_input", "allowed_decisions": [...]}]
  }
"""

import logging

# Configure DEBUG logging for the full vs-agentic-platform package tree.
# Each module uses logging.getLogger(__name__) — setting the root logger here
# ensures DEBUG messages from agent/, core/, and middleware/ all flow through.
# Change to logging.INFO in production to reduce log volume.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import build_agent, AgentContext

log = logging.getLogger(__name__)


# ── HITL handler ───────────────────────────────────────────────────────────────

def handle_ask_user_input(action_request: dict) -> Command:
    """
    Console handler for ask_user_input interrupts.

    Displays the LLM-generated question + options, gets the human's answer,
    and resumes with 'edit' to inject user_answer into the tool args.
    The tool then returns user_answer to the LLM as a ToolMessage.

    Resume format (identical to notebook 3.3):
      Command(resume={"decisions": [{"type": "edit", "edited_action": {...}}]})
    """
    args           = action_request["args"]
    question       = args["question"]
    options        = args["options"]
    allow_freetext = args.get("allow_freetext", True)

    print("\n" + "─" * 60)
    print(f"  {question}\n")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if allow_freetext:
        print("  (or type your own answer)")
    print("─" * 60)

    raw = input("  Your answer (number or text): ").strip()

    # Resolve number → option text
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        user_answer = options[int(raw) - 1]
    else:
        user_answer = raw

    print(f"  ✓ {user_answer}")

    return Command(
        resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "ask_user_input",
                    "args": {
                        "question":       question,
                        "options":        options,
                        "allow_freetext": allow_freetext,
                        "user_answer":    user_answer,  # ← injected
                    }
                }
            }]
        }
    )


# ── Main invocation loop ───────────────────────────────────────────────────────

def invoke_with_hitl(
    agent: Any,
    messages: list,
    config: dict,
    context: AgentContext,
) -> dict:
    """
    Invoke the agent with full dynamic HITL loop.

    Handles the pause/resume cycle for ask_user_input interrupts.
    All other tools (search, graph, summariser, chart) auto-execute and
    never cause an interrupt — this loop only ever handles ask_user_input.

    Args:
        agent:    Compiled LangChain agent from build_agent()
        messages: List of message dicts for the invocation
        config:   LangGraph config dict — must contain thread_id
        context:  AgentContext dict injected as runtime context

    Returns:
        Final agent response dict once no more interrupts remain.
    """
    current_input: Any = {"messages": messages}

    while True:
        response = agent.invoke(current_input, config=config, context=context)

        if not response.get("__interrupt__"):
            # No interrupt — agent finished and returned a final answer
            return response

        # Only one interrupt type possible: ask_user_input
        # (all other tools are not in interrupt_on)
        action_request = response["__interrupt__"][0].value["action_requests"][0]
        current_input  = handle_ask_user_input(action_request)


# ── Usage examples ─────────────────────────────────────────────────────────────

def run_examples():
    """
    Five example invocations covering all architectural paths:

    1. Normal request        — full pipeline, agent searches then answers
    2. Ambiguous request     — HITL fires: ask_user_input before search_tool
    3. Cache HIT             — same question again, skips all work
    4. PII in input          — PIIMiddleware redacts before LLM sees it
    5. Multi-turn            — episodic memory enriches second turn

    Thread-ID design
    ─────────────────
    thread_id = one conversation window. Same thread_id across turns = same
    conversation; LangGraph loads the checkpoint and appends. Different
    thread_id = different window, fresh checkpoint.

    Examples 1–4 each test an independent scenario and must NOT share
    message history — hence different thread_ids.

    Example 5 uses the SAME thread_id for both turns intentionally — that IS
    the multi-turn test. Turn 2's "the medication we just discussed" resolves
    only because LangGraph loads turn 1's checkpoint for the same thread.

    Why MemorySaver here (use_postgres=False, the default):
    The production FastAPI gateway calls build_agent(use_postgres=True) so HITL
    checkpoints survive pod restarts. run.py is a demo script — MemorySaver
    keeps checkpoints in process RAM and wipes them on exit. Re-running the
    script always starts clean, avoiding stale tool_call_id errors caused by
    old checkpoints accumulating in Postgres across repeated runs.
    """
    # use_postgres=False (default) → MemorySaver: checkpoints live in RAM,
    # wiped when this script exits. Safe to re-run without clearing Postgres.
    # In production, the gateway calls build_agent(use_postgres=True).
    agent = build_agent(domain="pharma")

    base_context: AgentContext = {
        "user_id":    "prudhvi_akella",
        "session_id": "demo_session_001",
        "domain":     "pharma",
    }

    # ── Example 1: Normal clinical query ──────────────────────────────────────
    print("\n" + "═" * 70)
    print("EXAMPLE 1 — Normal clinical query (full pipeline, no HITL)")
    print("═" * 70)
    result = invoke_with_hitl(
        agent,
        messages=[{"role": "user", "content": "What are the efficacy results for metformin in Type 2 diabetes?"}],
        config={"configurable": {"thread_id": "demo_session_001"}},
        context=base_context,
    )
    print(result["messages"][-1].content)

    # ── Example 2: Ambiguous query → HITL fires ────────────────────────────
    print("\n" + "═" * 70)
    print("EXAMPLE 2 — Ambiguous query (HITL: ask_user_input fires first)")
    print("═" * 70)
    result = invoke_with_hitl(
        agent,
        messages=[{"role": "user", "content": "Show me the trial data"}],
        config={"configurable": {"thread_id": "demo_session_002"}},
        context=base_context,
    )
    print(result["messages"][-1].content)

    # ── Example 3: Cache HIT ───────────────────────────────────────────────
    # Different thread from Example 1 — cache hit is SEMANTIC (Pinecone similarity),
    # not checkpoint-based. SemanticCacheMiddleware fires before the LLM regardless
    # of thread_id, as long as the question embedding is similar enough.
    print("\n" + "═" * 70)
    print("EXAMPLE 3 — Same question again (should be CACHE HIT)")
    print("═" * 70)
    result = invoke_with_hitl(
        agent,
        messages=[{"role": "user", "content": "What are the efficacy results for metformin in Type 2 diabetes?"}],
        config={"configurable": {"thread_id": "demo_session_003"}},
        context=base_context,
    )
    print(result["messages"][-1].content)

    # ── Example 4: PII in input ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("EXAMPLE 4 — PII in input (DomainPIIMiddleware redacts before LLM)")
    print("═" * 70)
    result = invoke_with_hitl(
        agent,
        messages=[{"role": "user", "content": "My email is patient@hospital.com. What are metformin contraindications?"}],
        config={"configurable": {"thread_id": "demo_session_004"}},
        context=base_context,
    )
    print(result["messages"][-1].content)

    # ── Example 5: Multi-turn ─────────────────────────────────────────────
    # Both turns share the SAME thread_id — intentional. LangGraph loads
    # turn 1's checkpoint when turn 2 arrives, so "the medication we just
    # discussed" resolves correctly. EpisodicMemoryMiddleware also surfaces
    # the turn-1 Q&A pair to enrich the system prompt for turn 2.
    print("\n" + "═" * 70)
    print("EXAMPLE 5 — Multi-turn (episodic memory enriches second turn)")
    print("═" * 70)
    mt_context: AgentContext = {
        "user_id":    "prudhvi_akella",
        "session_id": "demo_session_005",
        "domain":     "pharma",
    }
    mt_config = {"configurable": {"thread_id": "demo_session_005"}}   # shared across both turns

    invoke_with_hitl(
        agent,
        messages=[{"role": "user", "content": "What are the renal dosing guidelines for metformin?"}],
        config=mt_config,
        context=mt_context,
    )
    result = invoke_with_hitl(
        agent,
        messages=[{"role": "user", "content": "What about drug interactions for the medication we just discussed?"}],
        config=mt_config,
        context=mt_context,
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    # Credentials are fetched automatically from AWS at startup.
    # No environment variables needed for Pinecone or Postgres.
    #
    # Required AWS setup (one-time, per environment):
    #
    #   SSM Parameter Store (us-east-1 or your region):
    #     /clinical-agent/prod/pinecone/api_key     ← SecureString, KMS-encrypted
    #     /clinical-agent/prod/pinecone/index_name  ← String, e.g. "clinical-agent"
    #
    #   Secrets Manager:
    #     Secret name: clinical-agent/prod/postgres
    #     Secret value (JSON):
    #       {
    #         "host":     "your-rds-endpoint.rds.amazonaws.com",
    #         "port":     "5432",
    #         "dbname":   "clinical_agent",
    #         "username": "agent_user",
    #         "password": "..."
    #       }
    #
    #   IAM permissions (ECS task role / EC2 instance profile):
    #     ssm:GetParameter          on /clinical-agent/prod/*
    #     secretsmanager:GetSecretValue on clinical-agent/prod/*
    #     kms:Decrypt               on the KMS key used for SecureString params
    #
    # Always required (OpenAI — still from env var):
    #   export OPENAI_API_KEY="sk-..."
    #
    # Optional:
    #   export APP_ENV="staging"     # defaults to "prod"
    #   export LANGSMITH_TRACING="true"
    #   export LANGSMITH_API_KEY="ls__..."
    #
    # Pinecone index creation (run once per environment):
    #   from pinecone import Pinecone, ServerlessSpec
    #   pc = Pinecone(api_key="<from SSM>")
    #   pc.create_index("clinical-agent", dimension=1536, metric="cosine",
    #                   spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    run_examples()