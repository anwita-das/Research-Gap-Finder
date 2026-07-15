"""
temporal_agent.py

Phase 6 - Temporal Analysis Agent

Responsibilities:
- Build publication timelines
- Analyze concept evolution
- Analyze citation growth
- Detect emerging/declining concepts
- Generate temporal summary using LLM
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from src.gap_finding.graph_loader import GraphLoader
from src.gap_finding.paper_loader import PaperLoader
from src.gap_finding.llm_helper import LLMHelper


class TemporalAnalysisAgent:
    """
    Temporal reasoning agent for research gap detection.
    """

    def __init__(
        self,
        graph_loader: GraphLoader,
        paper_loader: PaperLoader
    ):

        self.graph_loader = graph_loader
        self.paper_loader = paper_loader

        # Phase 6 LLM
        self.llm = LLMHelper()


    # =====================================================
    # MAIN ANALYSIS FUNCTION
    # =====================================================

    def analyze(
        self,
        candidate: dict
    ) -> dict:
        """
        Generate Temporal Summary for one candidate gap.

        Input:
            Candidate edge from Phase 5

        Output:
            temporal_analysis schema
        """

        candidate_id = candidate["gap_id"]

        entity = candidate["shared_entity"]


        # ---------------------------------------------
        # Find papers connected to shared entity
        # ---------------------------------------------

        paper_nodes = (
            self.graph_loader
            .get_connected_papers(entity)
        )


        papers = (
            self.paper_loader
            .load_multiple(paper_nodes)
        )


        # ---------------------------------------------
        # Publication timeline
        # ---------------------------------------------

        years = [
            paper["year"]
            for paper in papers
            if paper.get("year", 0) > 0
        ]


        publication_counts = dict(
            Counter(years)
        )


        if years:

            first_year = min(years)
            latest_year = max(years)

        else:

            first_year = 0
            latest_year = 0



        # ---------------------------------------------
        # Trend detection
        # ---------------------------------------------

        trend = self._detect_trend(
            publication_counts
        )


        # ---------------------------------------------
        # Citation growth
        # ---------------------------------------------

        citation_growth = (
            self._calculate_citation_growth(
                papers
            )
        )


        # ---------------------------------------------
        # Emerging / declining entities
        # ---------------------------------------------

        emerging_entities = []
        declining_entities = []


        if trend == "Emerging":

            emerging_entities.append(entity)


        elif trend == "Declining":

            declining_entities.append(entity)



        # ---------------------------------------------
        # LLM generated summary
        # ---------------------------------------------

        trend_summary = (
            self._generate_summary(
                entity,
                publication_counts,
                first_year,
                latest_year,
                trend,
                citation_growth
            )
        )


        # ---------------------------------------------
        # Final schema
        # ---------------------------------------------

        return {

            "candidate_id": candidate_id,

            "entity": entity,

            "first_year": first_year,

            "latest_year": latest_year,

            "trend": trend,

            "publication_counts": publication_counts,

            "citation_growth": citation_growth,

            "emerging_entities": emerging_entities,

            "declining_entities": declining_entities,

            "trend_summary": trend_summary
        }



    # =====================================================
    # TREND DETECTION
    # =====================================================

    def _detect_trend(
    self,
    publication_counts: Dict[int, int]
) -> str:

        if len(publication_counts) < 3:
            return "Stable"


        years = sorted(publication_counts.keys())

        counts = [
            publication_counts[y]
            for y in years
        ]


        # Recent movement
        recent_change = counts[-1] - counts[-2]


        # Overall movement
        overall_change = counts[-1] - counts[0]


        # Peak detection
        peak_index = counts.index(max(counts))


        peak_is_recent = (
            peak_index >= len(counts) - 2
        )


        # Increasing toward recent years
        if recent_change > 0:
            return "Emerging"


        # Declined after reaching peak
        if (
            max(counts) > counts[0]
            and not peak_is_recent
            and recent_change < 0
        ):
            return "Declining"


        # Initial growth but now stable
        return "Stable"

    # =====================================================
    # CITATION GROWTH
    # =====================================================

    def _calculate_citation_growth(
        self,
        papers: List[dict]
    ) -> float:

        """
        Current implementation uses available
        citation_count metadata.

        Historical citation snapshots are not
        available yet, so this estimates growth.
        """


        total_citations = sum(

            paper.get(
                "citation_count",
                0
            )

            for paper in papers

        )


        if not papers:

            return 0.0


        return round(
            total_citations / len(papers),
            3
        )



    # =====================================================
    # LLM SUMMARY GENERATION
    # =====================================================

    def _generate_summary(
        self,
        entity,
        publication_counts,
        first_year,
        latest_year,
        trend,
        citation_growth
    ) -> str:


        prompt = f"""

You are a research trend analyst.

Analyze the evolution of this research concept.

Entity:
{entity}

Publication timeline:
{publication_counts}

First appearance:
{first_year}

Latest appearance:
{latest_year}

Detected trend:
{trend}

Average citation count indicator:
{citation_growth}


Generate a concise scientific temporal summary
(2-3 sentences).

Mention:
- how the research area evolved
- whether it is emerging, stable, or declining
- whether the field shows increasing activity, stability, or decline

Rules:
- Do not invent numbers.
- Only use provided information.
Do not interpret stable trends as growth.

Interpret the trend carefully:

- Increasing publications -> emerging/growing
- Similar publication counts -> stable/mature
- Decreasing recent publications -> declining/saturated

Do not call a decreasing trend growth.
"""


        try:

            return self.llm.generate(
                prompt
            )


        except Exception:

            return (
                f"{entity} shows a {trend.lower()} trend "
                f"from {first_year} to {latest_year}."
            )