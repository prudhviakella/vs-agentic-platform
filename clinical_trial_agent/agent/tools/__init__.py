"""
agent/tools/__init__.py
========================
Tool registry — import all tools and expose ALL_TOOLS list.

Tool categories:
  HITL gate (interrupt_on=True in middleware):
    ask_user_input  — single clarification gate, LLM calls when info is missing

  Auto-execute (not in interrupt_on, never pause):
    search_tool     — vector search over clinical knowledge base
    tavily_search   — live web search for current trials, approvals, news
    graph_tool      — biomedical knowledge graph queries
    summariser_tool — synthesise retrieved chunks
    chart_tool      — generate chart spec from numerical data

Constants:
  MAX_TOOL_CALLS_PER_REQUEST — enforced by ActionGuardrailMiddleware
  MAX_RETRIES_PER_TOOL       — retry cap per individual tool
"""

from langchain_community.tools.tavily_search import TavilySearchResults

from agent.tools.hitl import ask_user_input
from agent.tools.search import search_tool
from agent.tools.graph import graph_tool
from agent.tools.summariser import summariser_tool
from agent.tools.chart import chart_tool

MAX_TOOL_CALLS_PER_REQUEST = 5
MAX_RETRIES_PER_TOOL       = 3

# Live web search — LLM calls this for current trial data, recent approvals,
# drug safety updates, or anything that may have changed since training cutoff.
# The LLM knows Tavily well and will naturally call it after HITL clarification.
tavily_search = TavilySearchResults(
    max_results=5,
    description=(
        "Search the web for current clinical trial data, drug approvals, safety "
        "updates, and medical evidence. Use when the user asks about specific trials, "
        "drugs, conditions, or anything requiring up-to-date information. "
        "Call this after receiving clarification to retrieve relevant evidence."
    ),
)

# ask_user_input is first — it is the HITL gate (interrupt_on=True in middleware)
# All others auto-execute — not listed in interrupt_on, never cause a pause
ALL_TOOLS = [
    ask_user_input,
    search_tool,
    tavily_search,
    graph_tool,
    summariser_tool,
    chart_tool,
]

__all__ = [
    "ask_user_input",
    "search_tool",
    "tavily_search",
    "graph_tool",
    "summariser_tool",
    "chart_tool",
    "ALL_TOOLS",
    "MAX_TOOL_CALLS_PER_REQUEST",
    "MAX_RETRIES_PER_TOOL",
]