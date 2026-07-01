from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def clean_json_response(response: str) -> str:
    """Clean an LLM response so it can be parsed as JSON."""
    if not isinstance(response, str):
        return ""

    text = response.strip()
    if not text:
        return ""

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1, flags=re.IGNORECASE)
        text = text.strip()

    return text


def parse_llm_response(response: str) -> List[Dict[str, Any]]:
    """Parse an LLM response into a list of validated dictionaries."""
    cleaned = clean_json_response(response)
    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        payload = [payload]
    elif not isinstance(payload, list):
        return []

    validated: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        normalized = {
            "paper_id": item.get("paper_id", ""),
            "methods": item.get("methods", []) or [],
            "models": item.get("models", []) or [],
            "algorithms": item.get("algorithms", []) or [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "claims": [],
            "keywords": item.get("keywords", []) or [],
            "summary": item.get("summary", ""),
        }

        for key in [
            "methods",
            "models",
            "algorithms",
            "datasets",
            "tasks",
            "metrics",
            "claims",
            "keywords",
        ]:
            if not isinstance(normalized[key], list):
                normalized[key] = []

        if not isinstance(normalized["summary"], str):
            normalized["summary"] = ""

        validated.append(normalized)

    return validated
