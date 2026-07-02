from __future__ import annotations

from typing import Any, Dict, List


def _normalize_string_value(value: Any) -> str:
    """Convert a scalar or object-style entity into a clean string."""
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            return name.strip()

    return ""


def validate_list(values: Any) -> List[str]:
    """Validate and normalize a list of string values.

    Rules:
    - Must be a list
    - Remove empty strings
    - Remove duplicates
    - Strip whitespace
    - Keep only strings and simple object-style values
    - Reject null values
    """
    if not isinstance(values, list):
        return []

    cleaned: List[str] = []
    seen = set()

    for value in values:
        normalized = _normalize_string_value(value)
        if not normalized:
            continue

        key = normalized.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(normalized)

    return cleaned


def validate_extraction_result(result: Any) -> Dict[str, Any]:
    """Validate an extraction result and preserve the full shared schema."""
    if not isinstance(result, dict):
        result = {}

    validated = {
        "paper_id": result.get("paper_id", "") if isinstance(result.get("paper_id"), str) else "",
        "methods": validate_list(result.get("methods")),
        "models": validate_list(result.get("models")),
        "algorithms": validate_list(result.get("algorithms")),
        "datasets": validate_list(result.get("datasets")),
        "tasks": validate_list(result.get("tasks")),
        "metrics": validate_list(result.get("metrics")),
        "claims": validate_list(result.get("claims")),
        "keywords": validate_list(result.get("keywords")),
        "summary": result.get("summary", "") if isinstance(result.get("summary"), str) else "",
    }

    validated["paper_id"] = validated["paper_id"].strip()
    validated["summary"] = validated["summary"].strip()
    return validated
