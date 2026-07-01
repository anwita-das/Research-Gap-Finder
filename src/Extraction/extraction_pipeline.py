from __future__ import annotations

from typing import Any, Dict, Optional

from src.Extraction.entity_validator import validate_extraction_result
from src.Extraction.llm_parser import parse_llm_response
from src.Extraction.prompts import create_extraction_prompt


def extract_paper_entities(paper: Dict[str, Any], groq_client: Optional[Any] = None) -> Dict[str, Any]:
    """Extract research entities for a paper using the configured Groq client.

    The pipeline:
    1. Build a prompt from the paper metadata.
    2. Send it to Groq through the provided client.
    3. Parse the LLM response.
    4. Validate and normalize the result.

    Args:
        paper: A dictionary containing at least paper_id, title, and abstract.
        groq_client: A Groq client-like object with a generate(prompt: str) method.

    Returns:
        A schema-compatible extraction dictionary.
    """
    if not isinstance(paper, dict):
        paper = {}

    if groq_client is None:
        raise ValueError("groq_client is required")

    prompt = create_extraction_prompt(paper)

    try:
        response_text = groq_client.generate(prompt)
    except Exception as exc:
        print(f"Groq extraction failed: {exc}")
        response_text = ""

    parsed_results = parse_llm_response(response_text)
    if parsed_results:
        return validate_extraction_result(parsed_results[0])

    return validate_extraction_result({
        "paper_id": paper.get("paper_id", ""),
        "methods": [],
        "models": [],
        "algorithms": [],
        "datasets": [],
        "tasks": [],
        "metrics": [],
        "claims": [],
        "keywords": [],
        "summary": "",
    })
