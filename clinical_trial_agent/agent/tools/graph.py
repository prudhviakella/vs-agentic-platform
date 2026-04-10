"""
graph.py — graph_tool
======================
Biomedical knowledge graph queries against the Neo4j clinical trials database.
Auto-executes — not listed in interrupt_on, never causes a pause.

Graph schema (from clinical_trials_loader.py):
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
"""

import json
import logging
import os

from langchain_core.tools import tool

from agent.guardrails import sanitise_tool_results, validate_db_query

log = logging.getLogger(__name__)

# Cache the Neo4j driver — initialised once on first call
_driver = None


def _get_driver():
    """
    Return a connected Neo4j driver.
    Reads credentials from environment variables so secrets stay out of code.
    """
    global _driver
    if _driver is not None:
        return _driver

    from neo4j import GraphDatabase

    uri  = os.environ.get("NEO4J_URI",      "neo4j+s://52c31090.databases.neo4j.io")
    user = os.environ.get("NEO4J_USER",     "52c31090")
    pwd  = os.environ.get("NEO4J_PASSWORD", "")

    if not pwd:
        raise EnvironmentError(
            "NEO4J_PASSWORD not set. "
            "Add it to your environment or .env.local"
        )

    _driver = GraphDatabase.driver(uri, auth=(user, pwd))
    log.info(f"[GRAPH_TOOL] Connected to Neo4j  uri={uri}")
    return _driver


def _run_query(cypher: str, params: dict = None) -> list[dict]:
    """Run a read-only Cypher query and return results as a list of dicts."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


@tool(parse_docstring=True)
def graph_tool(entity: str) -> str:
    """
    Query the biomedical knowledge graph for clinical trial relationships.

    Use this tool to find:
    - Which trials study a specific drug or disease
    - Drug-drug interactions and contraindications
    - Trial sponsors and CROs
    - Countries/sites where a trial is conducted
    - MeSH terms and trial categories
    - Patient eligibility and outcomes for a trial

    Args:
        entity: Drug name, disease name, NCT ID, sponsor, or MeSH term to query.

    Returns:
        JSON string with trial relationships relevant to the entity.
    """
    ok, reason = validate_db_query(entity)
    if not ok:
        return json.dumps({"nodes": [], "edges": [], "error": reason})

    try:
        # Search across Trial, Drug, Disease, Sponsor, MeSHTerm using
        # case-insensitive CONTAINS so partial names work (e.g. "Pfizer" finds
        # "Pfizer Inc" and "BioNTech/Pfizer")
        cypher = """
        MATCH (t:Trial)
        WHERE t.briefTitle      CONTAINS $entity
           OR t.officialTitle   CONTAINS $entity
           OR t.nctId           = $entity
        WITH t LIMIT 5
        OPTIONAL MATCH (t)-[:TARGETS]->(d:Disease)
        OPTIONAL MATCH (t)-[:USES]->(dr:Drug)
        OPTIONAL MATCH (t)-[:SPONSORED_BY]->(s:Sponsor)
        OPTIONAL MATCH (t)-[:CONDUCTED_IN]->(co:Country)
        OPTIONAL MATCH (t)-[:ASSOCIATED_WITH]->(m:MeSHTerm)
        OPTIONAL MATCH (t)-[:MEASURES]->(o:Outcome {type: 'primary'})
        RETURN
            t.nctId          AS nctId,
            t.briefTitle     AS title,
            t.overallStatus  AS status,
            t.phase          AS phase,
            t.enrollmentCount AS enrollment,
            collect(DISTINCT d.name)  AS diseases,
            collect(DISTINCT dr.name) AS drugs,
            collect(DISTINCT s.name)  AS sponsors,
            collect(DISTINCT co.name) AS countries,
            collect(DISTINCT m.term)  AS meshTerms,
            collect(DISTINCT o.measure) AS primaryOutcomes
        """

        rows = _run_query(cypher, {"entity": entity})

        # If no trial found by title/NCT, search by drug or disease name
        if not rows:
            cypher2 = """
            MATCH (t:Trial)-[:USES|TARGETS|ASSOCIATED_WITH|SPONSORED_BY]->(n)
            WHERE toLower(n.name) CONTAINS toLower($entity)
               OR toLower(n.term) CONTAINS toLower($entity)
            WITH t LIMIT 5
            OPTIONAL MATCH (t)-[:TARGETS]->(d:Disease)
            OPTIONAL MATCH (t)-[:USES]->(dr:Drug)
            OPTIONAL MATCH (t)-[:SPONSORED_BY]->(s:Sponsor)
            OPTIONAL MATCH (t)-[:CONDUCTED_IN]->(co:Country)
            RETURN
                t.nctId         AS nctId,
                t.briefTitle    AS title,
                t.overallStatus AS status,
                t.phase         AS phase,
                t.enrollmentCount AS enrollment,
                collect(DISTINCT d.name)  AS diseases,
                collect(DISTINCT dr.name) AS drugs,
                collect(DISTINCT s.name)  AS sponsors,
                collect(DISTINCT co.name) AS countries
            """
            rows = _run_query(cypher2, {"entity": entity})

        if not rows:
            return json.dumps({
                "results": [],
                "message": f"No trials found for entity: '{entity}'",
            })

        # Format as readable chunks for the LLM
        chunks = []
        for row in rows:
            parts = [
                f"Trial: {row.get('nctId')} — {row.get('title')}",
                f"Status: {row.get('status')}  Phase: {row.get('phase')}  "
                f"Enrollment: {row.get('enrollment')}",
            ]
            if row.get("diseases"):
                parts.append(f"Diseases: {', '.join(row['diseases'])}")
            if row.get("drugs"):
                parts.append(f"Drugs: {', '.join(row['drugs'])}")
            if row.get("sponsors"):
                parts.append(f"Sponsors: {', '.join(row['sponsors'])}")
            if row.get("countries"):
                parts.append(f"Countries: {', '.join(row['countries'][:5])}")
            if row.get("meshTerms"):
                parts.append(f"MeSH: {', '.join(row['meshTerms'][:5])}")
            if row.get("primaryOutcomes"):
                parts.append(f"Primary outcomes: {'; '.join(row['primaryOutcomes'][:3])}")
            chunks.append("\n".join(parts))

        sanitised = sanitise_tool_results(chunks)
        log.info(
            f"[GRAPH_TOOL] {len(sanitised)} trial(s) found  entity='{entity[:50]}'"
        )
        return json.dumps({
            "results": sanitised,
            "source":  "neo4j/clinical-trials",
            "count":   len(sanitised),
        })

    except Exception as exc:
        log.error(f"[GRAPH_TOOL] Neo4j query failed: {exc}")
        return json.dumps({"results": [], "error": str(exc)})