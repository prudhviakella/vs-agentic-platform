"""
summariser.py — summariser_tool
=================================
Synthesises multiple retrieved chunks into one evidence-based summary.
Auto-executes — not listed in interrupt_on, never causes a pause.

WHY sync invoke() not ainvoke():
  LangChain's tool_node calls tools synchronously via a thread executor.
  async def tools raise NotImplementedError on sync invocation.
"""

import json
import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)


@tool(parse_docstring=True)
def summariser_tool(chunks_json: str) -> str:
    """
    Synthesise multiple retrieved text chunks into one coherent, evidence-based summary.

    Use this tool AFTER search_tool and graph_tool have returned results.
    Pass all retrieved chunks together so the summary covers all evidence.
    Do NOT call this before gathering evidence — it needs data to synthesise.

    Args:
        chunks_json: JSON string with key 'chunks' (list of retrieved text strings).

    Returns:
        A concise, factual synthesis of all provided evidence chunks.
        Falls back to concatenated raw chunks if synthesis fails.
    """
    try:
        data   = json.loads(chunks_json)
        chunks = data.get("chunks", [chunks_json])
    except Exception:
        chunks = [chunks_json]

    if not chunks:
        return "No content provided to summarise."

    try:
        llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            "Synthesise the following retrieved medical information into a clear, "
            "structured, evidence-based summary. Preserve all key numbers, study "
            "names, confidence levels, and clinical recommendations:\n\n"
            + "\n\n---\n\n".join(chunks[:5])
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        result   = response.content.strip()
        log.info(f"[SUMMARISER_TOOL] Synthesised {len(chunks)} chunks")
        return result
    except Exception as exc:
        log.warning(f"[SUMMARISER_TOOL] LLM failed → raw chunks fallback: {exc}")
        return "\n\n".join(chunks)
