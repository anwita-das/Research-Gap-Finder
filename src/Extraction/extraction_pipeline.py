from __future__ import annotations

from typing import Any, Dict, Optional

from src.entity_extraction.entity_extractor import EntityExtractor


def extract_paper_entities(
    paper: Dict[str, Any],
    groq_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Thin wrapper around the unified EntityExtractor interface."""
    if not isinstance(paper, dict):
        paper = {}

    extractor = EntityExtractor(groq_client=groq_client)
    return extractor.extract_entities(paper)
