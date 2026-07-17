"""
workflow/state.py

Shared workflow state for the Google ADK Gap Finding workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.gap_finding.schemas import (
    CandidateEdge,
    PaperContext,
    PaperSections,
    PaperSummary,
)


@dataclass
class GapFindingState:
    """
    Shared state passed between ADK agents.
    """

    # Current candidate
    candidate: CandidateEdge | None = None

    # Retrieved context
    context: PaperContext | None = None

    # Raw paper text
    source_text: str = ""
    target_text: str = ""

    # Extracted sections
    source_sections: PaperSections | None = None
    target_sections: PaperSections | None = None

    # Enriched summaries
    source_summary: PaperSummary | None = None
    target_summary: PaperSummary | None = None

    # Temporal reasoning output
    temporal_summary: dict | None = None

    # Comparison output
    comparison_summary: dict | None = None

    # Final detected gap
    detected_gap: dict | None = None

    # All completed gaps
    results: list[dict] = field(default_factory=list)