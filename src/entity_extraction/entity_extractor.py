"""Entity extraction pipeline for research papers using Groq."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Extraction.entity_validator import validate_extraction_result
from src.Extraction.groq_client import GroqClient
from src.Extraction.llm_parser import parse_llm_response
from src.Extraction.prompts import create_extraction_prompt

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EntityExtractor:
    """Extract datasets, tasks, metrics, and claims from a paper."""

    def __init__(self, groq_client: Optional[GroqClient] = None) -> None:
        """Initialize the extractor."""
        self.groq_client = groq_client or GroqClient()
        self.prompt_path = self._resolve_prompt_path()
        logger.debug("Initialized EntityExtractor with prompt template %s", self.prompt_path)

    def _resolve_prompt_path(self) -> Path:
        """Return the repository prompt template path."""
        return Path(__file__).resolve().parents[2] / "prompts" / "entity_extraction.md"

    def build_prompt(self, paper):
        return create_extraction_prompt(paper)

    def call_llm(self, prompt: str) -> str:
        """Send the prompt to the configured Groq client."""
        try:
            return self.groq_client.generate(prompt)
        except Exception as exc:
            logger.error("Groq request failed: %s", exc)
            return ""

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the raw LLM response into a normalized dictionary."""
        parsed_results = parse_llm_response(response_text)
        if parsed_results:
            return parsed_results[0]
        return {}

    def extract_entities(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from one paper and return the validated schema."""
        if not isinstance(paper, dict):
            raise TypeError("paper must be a dictionary")

        prompt = self.build_prompt(paper)
        
        response_text = self.call_llm(prompt)
        parsed = self.parse_response(response_text)

        raw_result = dict(parsed)

        # Always use the original paper_id from the source metadata rather than
        # trusting the LLM output, which may omit or alter it.
        raw_result["paper_id"] = str(paper.get("paper_id", "")).strip()

        raw_result.setdefault("keywords", [])
        raw_result.setdefault("summary", "")

        extracted = validate_extraction_result(raw_result)
        logger.info("Extracted entities for paper %s", extracted["paper_id"] or paper.get("title", "unknown"))
        return extracted


def main() -> None:
    """Load a merged paper, run entity extraction, and print the result."""
    logger.info("Starting entity extraction demo")

    data_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "merged"

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        print("Data directory not found.")
        return

    paper_files = sorted(data_dir.glob("*.json"))

    if not paper_files:
        logger.error("No JSON files found in %s", data_dir)
        print("No JSON files found.")
        return

    paper_path = paper_files[0]
    logger.info("Loading paper from %s", paper_path)

    try:
        with paper_path.open("r", encoding="utf-8") as handle:
            papers = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Failed to load paper file %s: %s", paper_path, exc)
        print(f"Failed to load paper file: {exc}")
        return

    if not isinstance(papers, list) or not papers:
        logger.error("No papers available in %s", paper_path)
        print("No papers available in the selected file.")
        return

    output_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "entities"
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = EntityExtractor()

    try:
        for paper in papers:
            if not isinstance(paper, dict):
                logger.warning("Skipping invalid paper object.")
                continue

            result = extractor.extract_entities(paper)

            output_file = output_dir / f"{result['paper_id']}.json"

            with output_file.open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2, ensure_ascii=False)

            logger.info("Saved extracted entities to %s", output_file)

        print(f"Successfully processed {len(papers)} papers.")

    except Exception as exc:  # pragma: no cover - defensive entry point
        logger.exception("Entity extraction failed: %s", exc)
        print(f"Entity extraction failed: {exc}")
        return

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
