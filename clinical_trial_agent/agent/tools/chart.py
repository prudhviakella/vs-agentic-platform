"""
chart.py — chart_tool
======================
Generates a chart specification from numerical data points.
Auto-executes — not listed in interrupt_on, never causes a pause.

Action guardrail: minimum 3 data points required (slide 9).
Production: pass chart_spec to matplotlib/plotly renderer.
"""

import json
import logging

from langchain_core.tools import tool

log = logging.getLogger(__name__)


@tool(parse_docstring=True)
def chart_tool(data_json: str) -> str:
    """
    Generate a chart specification from numerical data points.

    Only call this tool when you have 3 or more quantitative data points that
    would be meaningfully visualised (e.g., efficacy comparison across doses,
    time-series safety data, or comparative study results).
    Do NOT call this for qualitative or single-value responses.

    Args:
        data_json: JSON string with key 'data_points' containing a list of
                   objects each with keys 'label' (str) and 'value' (number).

    Returns:
        JSON string with chart specification, or explanation of why no chart was generated.
    """
    try:
        data        = json.loads(data_json)
        data_points = data.get("data_points", [])
    except Exception:
        return json.dumps({"chart": None, "reason": "Invalid JSON — could not parse data_points"})

    # Action guardrail: chart only if 3+ data points (slide 9)
    if len(data_points) < 3:
        return json.dumps({
            "chart":  None,
            "reason": f"Only {len(data_points)} data point(s). Minimum 3 required.",
        })

    chart_spec = {
        "type":        "bar",
        "title":       data.get("title", "Results"),
        "x_axis":      "Label",
        "y_axis":      "Value",
        "labels":      [str(dp.get("label", i)) for i, dp in enumerate(data_points)],
        "values":      [float(dp.get("value", 0)) for dp in data_points],
        "data_points": len(data_points),
    }
    log.info(f"[CHART_TOOL] Generated spec  points={len(data_points)}")
    return json.dumps({"chart": chart_spec, "renderer": "matplotlib"})
