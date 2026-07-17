"""
gap_detector_tool.py

Google ADK Gap Detector Tool.

Responsibilities
----------------
Prepare all evidence required by the
Gap Detector Agent.

This tool performs NO reasoning.

It simply packages:

- candidate edge
- enriched paper summaries
- temporal summary
- comparison summary

into one structured dictionary.
"""

from __future__ import annotations

from typing import Any


class GapDetectorTool:
    """
    Tool used by the Gap Detector Agent.

    It prepares all available evidence
    for LLM reasoning.
    """

    def prepare_gap_context(
        self,
        candidate: dict[str, Any],
        paper1: dict[str, Any],
        paper2: dict[str, Any],
        temporal_summary: dict[str, Any],
        comparison_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Combine all inputs into a single
        structured payload.
        """

        return {

            "candidate": candidate,

            "paper1": paper1,

            "paper2": paper2,

            "temporal_summary": temporal_summary,

            "comparison_summary": comparison_summary,

        }


# ---------------------------------------------------------
# ADK Tool Instance
# ---------------------------------------------------------

gap_detector_tool = GapDetectorTool()


# ---------------------------------------------------------
# ADK Callable Function
# ---------------------------------------------------------

def prepare_gap_context(
    candidate: dict[str, Any],
    paper1: dict[str, Any],
    paper2: dict[str, Any],
    temporal_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    ADK callable tool.

    Called by gap_detector_agent.
    """

    return gap_detector_tool.prepare_gap_context(
        candidate,
        paper1,
        paper2,
        temporal_summary,
        comparison_summary,
    )