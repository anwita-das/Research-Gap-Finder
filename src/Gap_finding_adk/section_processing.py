"""
section_processing.py

Deterministic preprocessing and postprocessing utilities
for the Google ADK Section Extraction workflow.

Responsibilities:
- Chunk paper text
- Initialize section storage
- Merge agent extraction outputs
- Remove duplicates
- Build final PaperSections schema

This file does NOT call any LLM.
"""

from __future__ import annotations

from .schemas import PaperSections


# ==========================================================
# Chunking
# ==========================================================

def chunk_text(
    paper_text: str,
    chunk_size: int = 8000,
    overlap: int = 500,
) -> list[str]:
    """
    Split a paper into overlapping chunks.

    Overlap prevents important information from being
    lost across chunk boundaries.
    """

    chunks = []

    start = 0

    while start < len(paper_text):

        end = min(
            start + chunk_size,
            len(paper_text),
        )

        chunks.append(
            paper_text[start:end]
        )

        if end == len(paper_text):
            break

        start = end - overlap

    return chunks


# ==========================================================
# Section Storage
# ==========================================================

def initialize_section_store() -> dict[str, list[str]]:
    """
    Create empty storage for extracted sections.
    """

    return {
        "methodology": [],
        "experimental_setup": [],
        "experimental_results": [],
        "discussion": [],
        "limitations": [],
        "future_work": [],
        "conclusion": [],
    }


# ==========================================================
# Merge Agent Outputs
# ==========================================================

def merge_sections(
    merged: dict[str, list[str]],
    extracted: dict,
) -> None:
    """
    Merge one ADK agent response into accumulated sections.

    The agent returns JSON:
    
    {
        "methodology": [],
        "experimental_setup": [],
        ...
    }
    """

    for key in merged:

        values = extracted.get(
            key,
            [],
        )

        if not isinstance(values, list):
            continue

        for value in values:

            if (
                isinstance(value, str)
                and value.strip()
            ):
                merged[key].append(
                    value.strip()
                )


# ==========================================================
# Deduplication
# ==========================================================

def deduplicate(
    values: list[str],
) -> list[str]:
    """
    Remove duplicate statements while preserving order.
    """

    seen = set()

    output = []

    for value in values:

        normalized = value.lower().strip()

        if normalized in seen:
            continue

        seen.add(normalized)

        output.append(value)

    return output


# ==========================================================
# Build Schema Object
# ==========================================================

def build_paper_sections(
    merged: dict[str, list[str]],
) -> PaperSections:
    """
    Convert extracted sections into PaperSections model.
    """

    return PaperSections(

        methodology=deduplicate(
            merged.get(
                "methodology",
                [],
            )
        ),

        experimental_setup=deduplicate(
            merged.get(
                "experimental_setup",
                [],
            )
        ),

        experimental_results=deduplicate(
            merged.get(
                "experimental_results",
                [],
            )
        ),

        discussion=deduplicate(
            merged.get(
                "discussion",
                [],
            )
        ),

        limitations=deduplicate(
            merged.get(
                "limitations",
                [],
            )
        ),

        future_work=deduplicate(
            merged.get(
                "future_work",
                [],
            )
        ),

        conclusion=deduplicate(
            merged.get(
                "conclusion",
                [],
            )
        ),
    )
