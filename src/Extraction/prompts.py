from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def create_extraction_prompt(paper: Dict[str, Any]) -> str:
    """Create a prompt that instructs Groq to extract the full entity schema.

    The prompt source of truth is the markdown template in the prompts directory.
    """
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "entity_extraction.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{{paper_id}}", str(paper.get("paper_id", "")).strip())
    prompt = prompt.replace("{{paper_title}}", str(paper.get("title", "")).strip())
    prompt = prompt.replace("{{paper_abstract}}", str(paper.get("abstract", "")).strip())
    return prompt