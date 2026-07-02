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
    """Parse an LLM response into a list of raw dictionaries."""
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

    parsed_results: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        normalized = {
            "paper_id": item.get("paper_id", "") if isinstance(item.get("paper_id"), str) else "",
            "methods": item.get("methods"),
            "models": item.get("models"),
            "algorithms": item.get("algorithms"),
            "datasets": item.get("datasets"),
            "tasks": item.get("tasks"),
            "metrics": item.get("metrics"),
            "claims": item.get("claims"),
            "keywords": item.get("keywords"),
            "summary": item.get("summary", "") if isinstance(item.get("summary"), str) else "",
        }

        parsed_results.append(normalized)

    return parsed_results
