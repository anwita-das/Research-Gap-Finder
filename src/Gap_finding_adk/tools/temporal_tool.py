"""
temporal_tool.py

Google ADK Temporal Tool.

Responsibilities
----------------
Compute deterministic temporal statistics used by
the Temporal Analysis Agent.

This tool performs NO LLM reasoning.

It computes:

- publication timeline
- publication counts
- trend heuristic
- citation statistics

The Temporal Agent interprets these statistics.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from gap_finding.graph_loader import GraphLoader
from gap_finding.paper_loader import PaperLoader

class TemporalTool:
    """
    Tool used by the Temporal Analysis Agent.

    Computes deterministic publication statistics.
    """

    def __init__(self):

        self.graph_loader = GraphLoader(
            "data/processed/knowledge_graph.json"
        )

        self.paper_loader = PaperLoader(
            "data/processed/enriched_papers"
        )

    # ---------------------------------------------------------

    def prepare_temporal_data(
        self,
        shared_entity: str,
    ) -> dict[str, Any]:
        """
        Build temporal statistics for one shared entity.
        """

        paper_nodes = self.graph_loader.get_connected_papers(
            shared_entity
        )

        papers = self.paper_loader.load_multiple(
            paper_nodes
        )

        years = [

            paper["year"]

            for paper in papers

            if paper.get("year", 0) > 0

        ]

        publication_counts = dict(
            Counter(years)
        )

        first_year = (
            min(years)
            if years
            else 0
        )

        latest_year = (
            max(years)
            if years
            else 0
        )

        trend = self._detect_trend(
            publication_counts
        )

        citation_growth = self._citation_growth(
            papers
        )

        return {

            "entity": shared_entity,

            "paper_count": len(papers),

            "publication_counts": publication_counts,

            "first_year": first_year,

            "latest_year": latest_year,

            "trend": trend,

            "citation_growth": citation_growth,

        }

    # ---------------------------------------------------------

    def _detect_trend(
        self,
        publication_counts: dict[int, int],
    ) -> str:
        """
        Simple deterministic trend heuristic.
        """

        if len(publication_counts) < 3:
            return "Stable"

        years = sorted(
            publication_counts.keys()
        )

        counts = [

            publication_counts[y]

            for y in years

        ]

        recent = counts[-1] - counts[-2]

        peak = counts.index(
            max(counts)
        )

        if recent > 0:
            return "Emerging"

        if (
            peak < len(counts) - 2
            and recent < 0
        ):
            return "Declining"

        return "Stable"

    # ---------------------------------------------------------

    def _citation_growth(
        self,
        papers: list[dict],
    ) -> float:
        """
        Average citation count.
        """

        if not papers:
            return 0.0

        total = sum(

            paper.get(
                "citation_count",
                0
            )

            for paper in papers

        )

        return round(
            total / len(papers),
            3,
        )


# ---------------------------------------------------------
# ADK Tool Instance
# ---------------------------------------------------------

temporal_tool = TemporalTool()


# ---------------------------------------------------------
# ADK Callable Function
# ---------------------------------------------------------

def prepare_temporal_data(
    shared_entity: str,
) -> dict[str, Any]:
    """
    ADK callable tool.

    Called by temporal_agent.
    """

    return temporal_tool.prepare_temporal_data(
        shared_entity
    )