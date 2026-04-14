"""
graph.py — graph_tool
======================
Cypher query executor against the Neo4j clinical trials graph.
The LLM generates the Cypher query — this tool just executes it.

Graph schema (loaded by clinical_trials_loader.py):
  Trial ──TARGETS──────► Disease
        ──USES──────────► Drug
        ──SPONSORED_BY──► Sponsor
        ──MANAGED_BY────► CRO
        ──CONDUCTED_IN──► Country
        ──LOCATED_AT────► Site ──IN_COUNTRY──► Country
        ──MEASURES──────► Outcome
        ──INCLUDES──────► PatientPopulation
        ──ASSOCIATED_WITH► MeSHTerm
        ──BELONGS_TO────► TrialCategory

Node properties:
  Trial:    nctId, briefTitle, officialTitle, phase, overallStatus,
            enrollmentCount, startDate, primaryCompletionDate
  Drug:     name, type, otherNames
  Disease:  name
  Sponsor:  name
  Country:  name
  Outcome:  measure, timeFrame, type (primary/secondary)
  MeSHTerm: term
  PatientPopulation: minimumAge, maximumAge, gender, eligibilityCriteria

Available trials include: Pfizer BNT162b2 (NCT04368728), Moderna mRNA-1273 (NCT04470427),
Remdesivir COVID (NCT04280705), Janssen Ad26.COV2.S (NCT04652245),
Hepatitis B TAF (NCT03753074), Heart Failure (NCT03164772),
Alzheimers (NCT03548935), Melanoma immunotherapy (NCT03518606),
Breast Cancer (NCT03434379), and more.
"""

import json
import logging
import os

from langchain_core.tools import tool

from agent.guardrails import sanitise_tool_results

log = logging.getLogger(__name__)

_driver = None


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver
    from neo4j import GraphDatabase
    uri  = os.environ.get("NEO4J_URI",      "neo4j+s://52c31090.databases.neo4j.io")
    user = os.environ.get("NEO4J_USER",     "52c31090")
    pwd  = os.environ.get("NEO4J_PASSWORD", "")
    if not pwd:
        raise EnvironmentError("NEO4J_PASSWORD not set")
    _driver = GraphDatabase.driver(uri, auth=(user, pwd))
    log.info(f"[GRAPH_TOOL] Connected to Neo4j  uri={uri}")
    return _driver


def _run_query(cypher: str, params: dict = None) -> list[dict]:
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


@tool(parse_docstring=True)
def graph_tool(cypher: str) -> str:
    """
    Execute a Cypher query against the Neo4j clinical trials knowledge graph.

    Use this tool to answer questions about:
    - Which trials study a specific drug or disease
    - Drug safety for a patient condition (contraindications, interactions)
    - Trial sponsors, CROs, and sites
    - Patient eligibility criteria (age, gender, health conditions)
    - Trial outcomes and endpoints
    - MeSH term classifications
    - Relationships between drugs, diseases, and trials

    GRAPH SCHEMA:
      (Trial)-[:TARGETS]->(Disease)
      (Trial)-[:USES]->(Drug)
      (Trial)-[:SPONSORED_BY]->(Sponsor)
      (Trial)-[:CONDUCTED_IN]->(Country)
      (Trial)-[:MEASURES]->(Outcome)
      (Trial)-[:INCLUDES]->(PatientPopulation)
      (Trial)-[:ASSOCIATED_WITH]->(MeSHTerm)

    EXAMPLE QUERIES:

    Find trials for a drug:
      MATCH (t:Trial)-[:USES]->(dr:Drug)
      WHERE toLower(dr.name) CONTAINS toLower('mRNA-1273')
      RETURN t.nctId, t.briefTitle, t.phase, t.overallStatus

    Find drug safety/contraindications for a condition:
      MATCH (t:Trial)-[:USES]->(dr:Drug)
      WHERE toLower(dr.name) CONTAINS toLower('mRNA-1273')
      MATCH (t)-[:TARGETS]->(d:Disease)
      MATCH (t)-[:INCLUDES]->(pp:PatientPopulation)
      RETURN t.nctId, t.briefTitle, dr.name, d.name,
             pp.minimumAge, pp.maximumAge, pp.eligibilityCriteria LIMIT 5

    Find trials by disease:
      MATCH (t:Trial)-[:TARGETS]->(d:Disease)
      WHERE toLower(d.name) CONTAINS toLower('heart failure')
      RETURN t.nctId, t.briefTitle, t.phase, t.overallStatus

    Find sponsor for a trial:
      MATCH (t:Trial)-[:SPONSORED_BY]->(s:Sponsor)
      WHERE t.nctId = 'NCT04470427'
      RETURN t.briefTitle, s.name

    IMPORTANT RULES:
    - Always use toLower() for case-insensitive string matching
    - Always LIMIT results (maximum 10)
    - Never use DELETE, CREATE, MERGE, SET — read-only queries only
    - Use CONTAINS for partial name matches, = for exact NCT IDs

    Args:
        cypher: A valid read-only Cypher query based on the schema above.

    Returns:
        JSON string with query results from the clinical trials graph.
    """
    # Safety: block write operations
    cypher_upper = cypher.strip().upper()
    forbidden = ["DELETE", "DETACH", "CREATE", "MERGE", "SET", "REMOVE", "DROP"]
    for keyword in forbidden:
        if keyword in cypher_upper:
            log.warning(f"[GRAPH_TOOL] Blocked write operation: {keyword}")
            return json.dumps({"error": f"Write operation '{keyword}' not allowed"})

    try:
        rows = _run_query(cypher)

        if not rows:
            return json.dumps({
                "results": [],
                "message": "No results found for this query.",
            })

        # Format as readable text chunks for the LLM
        chunks = []
        for row in rows:
            parts = []
            for key, value in row.items():
                if value is not None and value != [] and value != "":
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value if v)
                    parts.append(f"{key}: {value}")
            if parts:
                chunks.append("\n".join(parts))

        sanitised = sanitise_tool_results(chunks)
        log.info(
            f"[GRAPH_TOOL] {len(sanitised)} result(s) returned  "
            f"cypher='{cypher[:60]}...'"
        )
        return json.dumps({
            "results": sanitised,
            "source":  "neo4j/clinical-trials",
            "count":   len(sanitised),
        })

    except Exception as exc:
        log.error(f"[GRAPH_TOOL] Neo4j query failed: {exc}")
        return json.dumps({"results": [], "error": str(exc)})