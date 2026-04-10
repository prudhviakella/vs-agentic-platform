"""
search.py — search_tool
========================
Vector similarity search over the clinical-trials Pinecone index.
Auto-executes — not listed in interrupt_on, never causes a pause.

Index: clinical-trials-index  (namespace: clinical-trials)
Chunks contain: text, breadcrumbs, key_phrases, char_count, pii_redacted
"""

import json
import logging
import os

from langchain_core.tools import tool

from agent.guardrails import sanitise_tool_results, validate_db_query

log = logging.getLogger(__name__)

# Cache the Pinecone index connection — initialised once on first call
_index = None

def _get_index():
    """
    Return a connected Pinecone index for the clinical-trials namespace.
    Uses the same credentials as the rest of the platform (PINECONE_API_KEY).
    Index name is read from CLINICAL_TRIALS_INDEX env var, defaulting to
    'clinical-trials-index' which matches the console screenshot.
    """
    global _index
    if _index is not None:
        return _index

    from pinecone import Pinecone

    api_key    = os.environ.get("PINECONE_API_KEY", "")
    index_name = os.environ.get("CLINICAL_TRIALS_INDEX", "clinical-trials-index")

    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set")

    _index = Pinecone(api_key=api_key).Index(index_name)
    log.info(f"[SEARCH_TOOL] Connected to Pinecone index='{index_name}'")
    return _index


def _embed(text: str) -> list[float]:
    """
    Embed the query using the same model used at ingestion time.
    text-embedding-3-small (1536-dim) matches the index dimension.
    """
    from openai import OpenAI
    client = OpenAI()
    resp   = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


@tool(parse_docstring=True)
def search_tool(query: str) -> str:
    """
    Search clinical trial data using vector similarity search over Pinecone.

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

    try:
        index     = _get_index()
        embedding = _embed(query)

        results = index.query(
            vector=embedding,
            top_k=5,
            namespace="clinical-trials",
            include_metadata=True,
        )

        chunks = []
        for match in results.get("matches", []):
            meta  = match.get("metadata", {})
            text  = meta.get("text", "").strip()
            crumb = meta.get("breadcrumbs", "")
            score = round(match.get("score", 0.0), 4)

            if not text:
                continue

            chunks.append(
                f"[Source: {crumb} | score={score}]\n{text}"
            )

        if not chunks:
            log.info(f"[SEARCH_TOOL] No results  query='{query[:50]}'")
            return json.dumps({"results": [], "source": "pinecone", "top_k": 0})

        sanitised = sanitise_tool_results(chunks)
        log.info(
            f"[SEARCH_TOOL] Returned {len(sanitised)} chunks  "
            f"query='{query[:50]}'"
        )
        return json.dumps({
            "results": sanitised,
            "source":  "pinecone/clinical-trials",
            "top_k":   len(sanitised),
        })

    except Exception as exc:
        log.error(f"[SEARCH_TOOL] Pinecone query failed: {exc}")
        return json.dumps({"results": [], "error": str(exc)})