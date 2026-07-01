from __future__ import annotations

from typing import Any, Dict, List


def validate_list(values: Any) -> List[str]:
    """Validate and normalize a list of string values.

    Rules:
    - Must be a list
    - Remove empty strings
    - Remove duplicates
    - Strip whitespace
    - Keep only strings
    - Reject null values
    """
    if not isinstance(values, list):
        return []

    cleaned: List[str] = []
    seen = set()

    for value in values:
        if value is None:
            continue
        if not isinstance(value, str):
            continue

        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned


def validate_extraction_result(result: Any) -> Dict[str, Any]:
    """Validate an extraction result and preserve the full schema.

    Only the methods, models, and algorithms fields are validated in this phase.
    Other fields are preserved with defaults when absent or invalid.
    """
    if not isinstance(result, dict):
        result = {}

    validated = {
        "paper_id": result.get("paper_id", "") if isinstance(result.get("paper_id"), str) else "",
        "methods": validate_list(result.get("methods")),
        "models": validate_list(result.get("models")),
        "algorithms": validate_list(result.get("algorithms")),
        "datasets": [],
        "tasks": [],
        "metrics": [],
        "claims": [],
        "keywords": validate_list(result.get("keywords")),
        "summary": result.get("summary", "") if isinstance(result.get("summary"), str) else "",
    }

    return validated
