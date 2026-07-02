"""Entity extraction pipeline for research papers using Groq."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EntityExtractor:
    """Extract datasets, tasks, metrics, and claims from a paper.

    The extractor reads the paper title and abstract, builds a prompt from the
    repository prompt template, sends it to Groq, parses the JSON response, and
    validates the extracted entities before returning them in the project's
    extracted paper schema.
    """

    def __init__(self, groq_client: Optional[GroqClient] = None) -> None:
        """Initialize the extractor.

        Args:
            groq_client: Optional GroqClient instance. If omitted, a new client
                is created automatically.
        """
        self.groq_client = groq_client or GroqClient()
        self.prompt_path = self._resolve_prompt_path()
        logger.debug("Initialized EntityExtractor with prompt template %s", self.prompt_path)

    def _resolve_prompt_path(self) -> Path:
        """Return the repository prompt template path."""
        return Path(__file__).resolve().parents[2] / "prompts" / "entity_extraction.md"

    def build_prompt(self, paper: Dict[str, Any]) -> str:
        """Build a prompt for the given paper from the prompt template.

        Args:
            paper: A paper dictionary containing at least title and abstract.

        Returns:
            A populated prompt string for Groq.
        """
        if not isinstance(paper, dict):
            raise TypeError("paper must be a dictionary")

        title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        if not title and not abstract:
            raise ValueError("Paper must contain at least a title or abstract.")

        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_path}")

        prompt_template = self.prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace("{{paper_title}}", title)
        prompt = prompt.replace("{{paper_abstract}}", abstract)
        logger.debug("Built prompt for paper %s", paper.get("paper_id", "unknown"))
        return prompt

    def call_llm(self, prompt: str) -> str:
        """Send the prompt to the configured Groq client.

        Args:
            prompt: The fully built prompt string.

        Returns:
            The raw text response returned by Groq.
        """
        try:
            return self.groq_client.generate(prompt)
        except Exception as exc:
            logger.error("Groq request failed: %s", exc)
            return ""

    def parse_response(self, response_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse the raw LLM response into a JSON object.

        Args:
            response_text: Raw response from the LLM.

        Returns:
            A dictionary of entity lists, or empty lists if parsing fails.
        """
        if not response_text or not response_text.strip():
            logger.warning("Received empty LLM response")
            return {"datasets": [], "tasks": [], "metrics": [], "claims": []}

        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse LLM response: %s", exc)
            return {"datasets": [], "tasks": [], "metrics": [], "claims": []}

        if not isinstance(parsed, dict):
            logger.warning("LLM response was not a JSON object")
            return {"datasets": [], "tasks": [], "metrics": [], "claims": []}

        return {
            "datasets": parsed.get("datasets") if isinstance(parsed.get("datasets"), list) else [],
            "tasks": parsed.get("tasks") if isinstance(parsed.get("tasks"), list) else [],
            "metrics": parsed.get("metrics") if isinstance(parsed.get("metrics"), list) else [],
            "claims": parsed.get("claims") if isinstance(parsed.get("claims"), list) else [],
        }

    def validate_entities(self, raw_entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize a list of extracted entities.

        Args:
            raw_entities: Entity objects returned by the model.

        Returns:
            A cleaned list of valid entities without duplicates.
        """
        validated: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for item in raw_entities:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            confidence_raw = item.get("confidence", 0.0)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                continue

            if not name or not 0.0 <= confidence <= 1.0:
                continue

            normalized_name = name.strip()
            if normalized_name.lower() in seen:
                continue

            seen.add(normalized_name.lower())
            validated.append({"name": normalized_name, "confidence": confidence})

        return validated

    def extract_entities(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from one paper and return the validated schema.

        Args:
            paper: A paper dictionary that follows the project's master schema.

        Returns:
            A dictionary in the extracted paper schema format.
        """
        if not isinstance(paper, dict):
            raise TypeError("paper must be a dictionary")

        prompt = self.build_prompt(paper)
        response_text = self.call_llm(prompt)
        parsed = self.parse_response(response_text)

        keywords = paper.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        extracted = {
            "paper_id": str(paper.get("paper_id", "")).strip(),
            "methods": [],
            "models": [],
            "algorithms": [],
            "datasets": self.validate_entities(parsed.get("datasets", [])),
            "tasks": self.validate_entities(parsed.get("tasks", [])),
            "metrics": self.validate_entities(parsed.get("metrics", [])),
            "claims": self.validate_entities(parsed.get("claims", [])),
            "keywords": keywords,
            "summary": "",
        }

        logger.info("Extracted entities for paper %s", extracted["paper_id"] or paper.get("title", "unknown"))
        return extracted


def main() -> None:
    """Load a merged paper, run entity extraction, and print the result."""
    logger.info("Starting entity extraction demo")

    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "merged"
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        print("Data directory not found.")
        return

    paper_files = sorted(data_dir.glob("*.json"))
    if not paper_files:
        logger.error("No merged paper files found in %s", data_dir)
        print("No merged paper files found.")
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

    paper = papers[0]
    if not isinstance(paper, dict):
        logger.error("The first item in %s is not a paper object", paper_path)
        print("The selected input is not a valid paper object.")
        return

    try:
        extractor = EntityExtractor()
        result = extractor.extract_entities(paper)
        output_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "entities"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{result['paper_id']}.json"

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

        logger.info("Saved extracted entities to %s", output_file)
    except Exception as exc:  # pragma: no cover - defensive entry point
        logger.exception("Entity extraction failed: %s", exc)
        print(f"Entity extraction failed: {exc}")
        return

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
