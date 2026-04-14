"""
app.py — VS Agentic Platform · Chainlit UI
==========================================
Clinical Trial Research Assistant — full HITL support.

Run:
    pip install -r requirements.txt
    chainlit run app.py --port 8501

Environment:
    AGENT_API_URL   — FastAPI base URL   (default: http://localhost:8000)
    AGENT_API_KEY   — X-API-Key header   (default: local-dev-key)
    AGENT_DOMAIN    — domain             (default: pharma)
"""

import os
import re
import uuid
import httpx
import chainlit as cl

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("AGENT_API_KEY", "local-dev-key")
DOMAIN  = os.environ.get("AGENT_DOMAIN",  "pharma")
AGENT   = "clinical-trial"

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

STARTERS = [
    "What are the Phase 3 efficacy results for Pfizer BNT162b2?",
    "Tell me about the COVID vaccine trial",
    "Is mRNA-1273 safe for patients with heart failure?",
    "Which trials study remdesivir for COVID-19?",
    "What are the primary outcomes for the Moderna vaccine trial NCT04470427?",
    "Who sponsors the Hepatitis B TAF trial?",
]

# ── Answer cleanup helpers ───────────────────────────────────────────────────

def _clean_answer(answer: str) -> str:
    """
    Strip internal tags and duplicate disclaimers from the agent answer
    before showing it to the user.

    WHY strip EPISODIC tag:
      The middleware uses EPISODIC: YES/NO to decide whether to store the
      answer in episodic memory. It is an internal signal — not for users.
      A regex handles all spacing/case variations reliably.

    WHY strip duplicate disclaimer:
      The system prompt instructs the LLM to append the disclaimer.
      The UI footer also appends it. Stripping from the answer prevents
      the user seeing the same sentence twice.
    """
    # Remove EPISODIC tag in all variations
    answer = re.sub(r'\n?EPISODIC:\s*(YES|NO)\s*', '', answer, flags=re.IGNORECASE)

    # Remove duplicate disclaimer added by the LLM (footer will add it back)
    answer = re.sub(
        r'\n?This information is for research purposes only'
        r' and does not constitute medical advice\.?\s*',
        '',
        answer,
        flags=re.IGNORECASE,
    )

    # Remove internal guardrail reason if it somehow leaks through
    answer = re.sub(
        r'\n?\[Reason logged for review:.*?\]\s*',
        '',
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return answer.strip()


# ── Session helpers ──────────────────────────────────────────────────────────

def get_thread_id() -> str:
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())[:12]
        cl.user_session.set("thread_id", thread_id)
    return thread_id

def set_interrupted(val: bool):
    cl.user_session.set("interrupted", val)

def is_interrupted() -> bool:
    return bool(cl.user_session.get("interrupted", False))


# ── Lifecycle ────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    """Show welcome message with starter suggestions."""
    thread_id = get_thread_id()
    set_interrupted(False)

    actions = [
        cl.Action(name="starter", value=q, label=q, description=q, payload={"value": q})
        for q in STARTERS
    ]

    await cl.Message(
        content=(
            "## ⚕ Clinical Trial Research Assistant\n\n"
            "Explore clinical trial data, drug efficacy, safety profiles, "
            "and biomedical knowledge graphs powered by real Pinecone + Neo4j data.\n\n"
            f"**Session:** `{thread_id}`  ·  **Domain:** `{DOMAIN}`\n\n"
            "---\n\n**Try one of these questions:**"
        ),
        actions=actions,
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Route to chat or resume depending on HITL state."""
    thread_id = get_thread_id()
    if is_interrupted():
        await _resume(thread_id, message.content)
    else:
        await _chat(thread_id, message.content)


@cl.action_callback("starter")
async def on_starter(action: cl.Action):
    """Handle starter suggestion click."""
    await action.remove()
    thread_id = get_thread_id()
    await _chat(thread_id, action.payload["value"])


@cl.action_callback("hitl_option")
async def on_hitl_option(action: cl.Action):
    """Handle HITL option button click."""
    await action.remove()
    thread_id = get_thread_id()

    await cl.Message(
        content=f"✅ Selected: **{action.payload['value']}**",
        author="You",
    ).send()

    await _resume(thread_id, action.payload["value"])


# ── API calls ────────────────────────────────────────────────────────────────

async def _chat(thread_id: str, text: str):
    """Send a new message to the agent."""
    async with cl.Step(name="🔍 Searching knowledge base", show_input=False) as step:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{API_URL}/api/v1/{AGENT}/chat",
                    headers=HEADERS,
                    json={"message": text, "thread_id": thread_id, "domain": DOMAIN},
                )
            resp.raise_for_status()
            data = resp.json()
            step.output = f"Latency: {data.get('latency_ms', 0):.0f}ms"
        except httpx.ConnectError:
            await cl.Message(
                content=(
                    "❌ **Cannot connect to the agent API.**\n\n"
                    f"Make sure the FastAPI server is running at `{API_URL}`\n\n"
                    "```bash\nuvicorn vs_platform.main:app --host 0.0.0.0 --port 8000\n```"
                )
            ).send()
            return
        except Exception as exc:
            await cl.Message(content=f"❌ Error: {exc}").send()
            return

    await _handle_response(data)


async def _resume(thread_id: str, user_answer: str):
    """Resume a paused HITL conversation."""
    async with cl.Step(name=f"🔍 Searching for: {user_answer[:40]}", show_input=False) as step:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{API_URL}/api/v1/{AGENT}/resume",
                    headers=HEADERS,
                    json={"thread_id": thread_id, "user_answer": user_answer, "domain": DOMAIN},
                )
            resp.raise_for_status()
            data = resp.json()
            step.output = f"Latency: {data.get('latency_ms', 0):.0f}ms"
        except Exception as exc:
            set_interrupted(False)
            await cl.Message(content=f"❌ Resume error: {exc}").send()
            return

    await _handle_response(data)


async def _handle_response(data: dict):
    """Route API response to HITL question or final answer."""

    if data.get("interrupted"):
        payload  = data.get("interrupt_payload", {})
        question = payload.get("question", "Please clarify:")
        options  = payload.get("options", [])
        allow_ft = payload.get("allow_freetext", True)

        set_interrupted(True)

        # Numbered options displayed in message body
        options_md = "\n".join(
            f"> **{i+1}.** {opt}" for i, opt in enumerate(options)
        )

        # Action buttons for one-click selection
        actions = [
            cl.Action(
                name="hitl_option",
                label=f"  {i+1}. {opt}  ",
                description=opt,
                payload={"value": opt},
            )
            for i, opt in enumerate(options)
        ]

        hint = "\n\n_Or type a custom answer in the chat input below._" if allow_ft else ""
        await cl.Message(
            content=(
                f"### 🔍 Clarification needed\n\n"
                f"**{question}**\n\n"
                f"{options_md}"
                f"{hint}\n\n"
                f"**Click an option below or type your own:**"
            ),
            actions=actions,
        ).send()

    else:
        set_interrupted(False)
        answer = (data.get("answer") or "").strip()
        latency = data.get("latency_ms", 0)

        if not answer:
            await cl.Message(
                content=(
                    "⚠️ No answer returned. The response may not have met "
                    "safety standards. Try rephrasing your question."
                )
            ).send()
            return

        # Clean answer — strip internal tags and duplicate disclaimers
        answer = _clean_answer(answer)

        footer = (
            f"\n\n---\n"
            f"*⏱ {latency/1000:.1f}s · "
            f"This information is for research purposes only "
            f"and does not constitute medical advice.*"
        )

        await cl.Message(
            content=answer + footer,
            author="Clinical Trial Assistant",
        ).send()