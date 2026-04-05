"""
agent/__init__.py — Public API
================================
Re-exports the public surface so all consumers use clean top-level imports:

    from agent import build_agent, AgentContext, ALL_TOOLS

This means no file outside the package ever needs to write
"from agent.agent import ..." or "from agent.schema import ..." — those
internal module paths stay as implementation details.
"""

from agent.agent import build_agent        # noqa: F401
from agent.schema import AgentContext      # noqa: F401
from agent.tools import ALL_TOOLS         # noqa: F401

__all__ = ["build_agent", "AgentContext", "ALL_TOOLS"]
