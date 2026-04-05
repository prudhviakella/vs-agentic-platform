"""
search.py — search_tool
========================
Vector similarity search over clinical trial and safety database.
Auto-executes — not listed in interrupt_on, never causes a pause.

Production: replace simulated results with Pinecone / pgvector client.
"""

import json
import logging

from langchain_core.tools import tool

from agent.guardrails import sanitise_tool_results, validate_db_query

log = logging.getLogger(__name__)


@tool(parse_docstring=True)
def search_tool(query: str) -> str:
    """
    Search clinical trial and safety database using vector similarity.

    Use this tool when the user asks about drug efficacy, clinical study results,
    safety profiles, adverse events, or evidence-based treatment recommendations.
    Always search before composing an answer — never rely on memory alone.

    Args:
        query: Concise search query (5-15 words). Be specific about the drug or condition.

    Returns:
        JSON string containing retrieved chunks from the clinical knowledge base.
    """
    ok, reason = validate_db_query(query)
    if not ok:
        return json.dumps({"results": [], "error": reason})

    # Production: replace with async Pinecone / pgvector client
    raw_results = [
        (
            f"[Source: clinical_trials_db] Query='{query}' | "
            "Phase 3 RCT (n=2,847): Primary endpoint reduction 42% vs placebo (p<0.001, CI 95%). "
            "Median OS improvement 4.2 months (HR 0.71)."
        ),
        (
            f"[Source: safety_db] Query='{query}' | "
            "Common AEs: nausea 12%, headache 8%, fatigue 6%. "
            "Serious AEs <2%. Contraindicated: severe renal impairment (eGFR < 30 mL/min)."
        ),
        (
            f"[Source: guidelines_db] Query='{query}' | "
            "ADA 2024 Clinical Practice Guidelines: first-line recommendation. "
            "Dose-adjust for renal function per eGFR banding. "
            "Monitor HbA1c every 3 months initially."
        ),
    ]

    sanitised = sanitise_tool_results(raw_results)
    log.info(f"[SEARCH_TOOL] Returned {len(sanitised)} chunks  query='{query[:50]}'")
    return json.dumps({"results": sanitised, "source": "vector_db", "top_k": len(sanitised)})
