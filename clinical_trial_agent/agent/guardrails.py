"""
guardrails.py — Pure Functions (stateless input → output, all < 1ms, no LLM)
=============================================================================
WHY pure functions here:
  - Easy to unit-test with zero mocking
  - No side effects — safe to call from gateway or agent middleware
  - Deterministic first (cheap) → model-based second (expensive) is the
    golden rule from slide 9 + the Guardrails doc

Ownership by caller:
  check_prompt_injection()   → GATEWAY (generic security, not domain-specific)
  check_toxic()              → AGENT middleware (domain rule — pharma toxic ≠ finance)
  check_medical_action_*()   → AGENT middleware (pharma-domain output rule)
  validate_db_query()        → AGENT tool (domain action guardrail)
  sanitise_tool_results()    → AGENT tool (retrieval sanitiser)
"""

import re

# ── Regex banks ────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"disregard\s+(the|your|all)\s+(above|previous|prior)",
    r"you\s+are\s+now\s+",
    r"\bact\s+as\b",
    r"pretend\s+(you\s+are|to\s+be)",
    r"\bjailbreak\b",
    r"<\|system\|>",
    r"\[system\]",
    r"new\s+instructions?:",
]

_TOXIC_PATTERNS = [
    r"\b(kill|murder|harm|attack)\b.{0,30}(person|people|patient|user)",
    r"how\s+to\s+(make|build|create).{0,20}(bomb|weapon|poison)",
]

# slide 10 — "Code catches known patterns instantly" (under 1ms, before LLM judge)
_MEDICAL_ACTION_PATTERNS = [
    r"you\s+should\s+take",
    r"stop\s+your\s+medication",
    r"dosage\s+is\s+\d+",
    r"take\s+\d+\s*mg",
    r"inject\s+yourself",
    r"patient\s+should\s+(stop|start|take|increase|decrease)",
    r"administer\s+\d+",
]


# ── Pure functions ─────────────────────────────────────────────────────────────

def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Pure. Returns (is_clean, reason). Regex only — no LLM."""
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, text.lower()):
            return False, f"Prompt injection detected: '{pat}'"
    return True, ""


def check_toxic(text: str) -> tuple[bool, str]:
    """Domain toxic check — pharma-specific patterns."""
    for pat in _TOXIC_PATTERNS:
        if re.search(pat, text.lower()):
            return False, "Toxic content pattern matched"
    return True, ""


def run_input_guardrails(text: str) -> tuple[bool, str]:
    """
    GATEWAY function — generic security checks only.
    Called by gateway.py BEFORE agent.run() is invoked.

    Only contains domain-agnostic checks:
      - Prompt injection (generic adversarial pattern — same for every domain)

    Does NOT contain:
      - PII detection    (domain-specific)
      - Toxic content    (domain-specific)
      - Out-of-domain    (domain-specific)
    """
    return check_prompt_injection(text)


def check_medical_action_output(answer: str) -> tuple[bool, str]:
    """
    Code-first output check (slide 10 — 'Detect: Code First').
    Runs BEFORE the expensive LLM faithfulness judge.
    WHY: regex is sub-millisecond; known dangerous directives must never
    reach the user regardless of LLM confidence score.
    """
    for pat in _MEDICAL_ACTION_PATTERNS:
        if re.search(pat, answer.lower()):
            return False, f"Medical action directive: '{pat}'"
    return True, ""


def validate_db_query(query: str) -> tuple[bool, str]:
    """
    Action guardrail — enforces read-only DB access (slide 9 layer 03).
    Pure function: no state, no side effects.
    """
    forbidden = {"insert", "update", "delete", "drop", "truncate", "alter", "create"}
    for word in forbidden:
        if re.search(rf"\b{word}\b", query.lower()):
            return False, f"Write op '{word}' blocked — read-only enforced"
    return True, ""


def sanitise_tool_results(chunks: list[str]) -> list[str]:
    """
    Retrieval Sanitiser (slide 9 footer note).
    Strips injection patterns from ALL tool results before agent sees them.
    WHY: external data sources (vector DBs, graph DBs) may contain injected
    instructions planted by adversaries. Pure defensive step.
    """
    sanitised = []
    for chunk in chunks:
        clean = chunk
        for pat in _INJECTION_PATTERNS:
            clean = re.sub(pat, "[REDACTED]", clean, flags=re.IGNORECASE)
        sanitised.append(clean)
    return sanitised


def count_tokens_approx(text: str) -> int:
    """1 token ≈ 4 chars — rough budget check, no tiktoken dependency."""
    return max(1, len(text) // 4)
