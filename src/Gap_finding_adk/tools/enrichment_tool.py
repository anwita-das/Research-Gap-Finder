"""
enrichment_tool.py

Google ADK Enrichment Tool.

Responsibilities
----------------
Prepare all structured information required by the
Enrichment Agent.

This tool performs NO LLM reasoning.

It simply combines:

- Paper metadata
- Existing entity extraction
- Extracted paper sections

into one payload for the ADK enrichment agent.
"""

from __future__ import annotations

from typing import Any

from Gap_finding_adk.schemas import PaperSections


class EnrichmentTool:
    """
    Tool used by the enrichment agent.

    Packages deterministic information before
    LLM-based enrichment.
    """

    def prepare_enrichment(
        self,
        paper: dict[str, Any],
        entities: dict[str, Any],
        sections: PaperSections,
    ) -> dict[str, Any]:
        """
        Prepare enrichment payload.

        Parameters
        ----------
        paper
            Original paper metadata.

        entities
            Existing entity extraction.

        sections
            Structured sections extracted by the
            Section Extraction Agent.

        Returns
        -------
        dict
            Payload consumed by enrichment_agent.
        """

        return {

            # -------------------------
            # Metadata
            # -------------------------

            "paper_id": paper.get("paper_id"),

            "title": paper.get("title"),

            "year": paper.get("year"),

            "authors": paper.get("authors", []),

            "abstract": paper.get("abstract", ""),

            # -------------------------
            # Existing extracted entities
            # -------------------------

            "methods": entities.get("methods", []),

            "models": entities.get("models", []),

            "algorithms": entities.get("algorithms", []),

            "datasets": entities.get("datasets", []),

            "tasks": entities.get("tasks", []),

            "metrics": entities.get("metrics", []),

            "claims": entities.get("claims", []),

            "keywords": entities.get("keywords", []),

            "summary": entities.get("summary", ""),

            # -------------------------
            # Extracted paper sections
            # -------------------------

            "sections": sections.model_dump(),
        }


# ---------------------------------------------------------
# ADK Tool Instance
# ---------------------------------------------------------

enrichment_tool = EnrichmentTool()


# ---------------------------------------------------------
# ADK Callable Function
# ---------------------------------------------------------

def prepare_enrichment(
    paper: dict[str, Any],
    entities: dict[str, Any],
    sections: PaperSections,
) -> dict[str, Any]:
    """
    ADK callable tool.

    Called by enrichment_agent.
    """

    return enrichment_tool.prepare_enrichment(
        paper,
        entities,
        sections,
    )