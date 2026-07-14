from __future__ import annotations

import json
import re

from .schemas import PaperSummary

class EnrichmentParseError(Exception):
    """Raised when the LLM response cannot be parsed."""
    pass

# ==========================================================
# Cleaning
# ==========================================================

def clean_json_response(response: str) -> str:
    """
    Cleans a raw LLM response so it can be parsed as JSON.
    """

    if not isinstance(response, str):
        return ""

    text = response.strip()

    if not text:
        return ""

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        text = match.group(1).strip()

    elif text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

        text = text.strip()

    return text

# ==========================================================
# Validation Helpers
# ==========================================================

def _validate_list(values):

    if not isinstance(values, list):
        return []

    cleaned = []
    seen = set()

    for value in values:

        if not isinstance(value, str):
            continue

        value = value.strip()

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(value)

    return cleaned


def _validate_string(value):

    if not isinstance(value, str):
        return ""

    return value.strip()

# ==========================================================
# Paper Enrichment Parser
# ==========================================================

def parse_enrichment_response(
    response: str,
    paper: dict,
    entities: dict,
) -> PaperSummary:

    cleaned = clean_json_response(response)

    if not cleaned:
        raise EnrichmentParseError(
            "Empty LLM response."
        )

    try:

        enrichment = json.loads(cleaned)

        if not isinstance(enrichment, dict):
            raise EnrichmentParseError(
                "LLM must return a JSON object."
            )

    except json.JSONDecodeError as e:

        raise EnrichmentParseError(
            f"Invalid JSON returned by LLM: {e}"
        )

    return PaperSummary(

        # ---------- Metadata ----------

        paper_id=paper["paper_id"],

        title=paper["title"],

        year=paper.get("year", 0),

        authors=paper.get("authors", []),

        abstract=paper.get("abstract", ""),

        # ---------- Existing entities ----------

        methods=_validate_list(
            entities.get("methods")
        ),

        models=_validate_list(
            entities.get("models")
        ),

        algorithms=_validate_list(
            entities.get("algorithms")
        ),

        datasets=_validate_list(
            entities.get("datasets")
        ),

        tasks=_validate_list(
            entities.get("tasks")
        ),

        metrics=_validate_list(
            entities.get("metrics")
        ),

        claims=_validate_list(
            entities.get("claims")
        ),

        keywords=_validate_list(
            entities.get("keywords")
        ),

        summary=_validate_string(
            entities.get("summary")
        ),

        # ---------- New enrichment ----------

        experimental_results=_validate_list(
            enrichment.get("experimental_results")
        ),

        limitations=_validate_list(
            enrichment.get("limitations")
        ),

        future_work=_validate_list(
            enrichment.get("future_work")
        ),

        key_contributions=_validate_list(
            enrichment.get("key_contributions")
        ),

        novelty_points=_validate_list(
            enrichment.get("novelty_points")
        ),

        experimental_setup=_validate_string(
            enrichment.get("experimental_setup")
        ),

        implementation_details=_validate_list(
            enrichment.get("implementation_details")
        ),

    )