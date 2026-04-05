"""
graph.py — graph_tool
======================
Biomedical knowledge graph query for entity relationships.
Auto-executes — not listed in interrupt_on, never causes a pause.

Production: replace simulated data with Neo4j Bolt or AWS Neptune client.
"""

import json
import logging

from langchain_core.tools import tool

from agent.guardrails import sanitise_tool_results, validate_db_query

log = logging.getLogger(__name__)


@tool(parse_docstring=True)
def graph_tool(entity: str) -> str:
    """
    Query the biomedical knowledge graph for entity relationships.

    Use this tool to find drug-drug interactions, contraindications, disease-drug
    connections, mechanism of action pathways, or any relationship-based queries
    that require traversing connected medical entities.

    Args:
        entity: The primary entity to query (drug name, disease, gene, protein).

    Returns:
        JSON string with graph nodes and edges relevant to the entity.
    """
    ok, reason = validate_db_query(entity)
    if not ok:
        return json.dumps({"nodes": [], "edges": [], "error": reason})

    # Production: replace with Neo4j Bolt or AWS Neptune client
    graph_data = {
        "query_entity": entity,
        "nodes": [
            {"id": "drug_001", "type": "Drug",      "name": entity,           "properties": {"class": "biguanide"}},
            {"id": "cond_001", "type": "Condition", "name": "Type 2 Diabetes","properties": {"icd10": "E11"}},
            {"id": "risk_001", "type": "Risk",      "name": "Renal Impairment","properties": {"eGFR_threshold": 30}},
            {"id": "drug_002", "type": "Drug",      "name": "Contrast Media", "properties": {"interaction_risk": "high"}},
        ],
        "edges": [
            {"from": "drug_001", "to": "cond_001", "relation": "TREATS",             "confidence": 0.98},
            {"from": "drug_001", "to": "risk_001", "relation": "CONTRAINDICATED_IN", "confidence": 0.95},
            {"from": "drug_001", "to": "drug_002", "relation": "INTERACTS_WITH",     "confidence": 0.88},
        ],
    }

    sanitised = sanitise_tool_results([json.dumps(graph_data)])[0]
    log.info(f"[GRAPH_TOOL] {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges  entity='{entity}'")
    return sanitised
