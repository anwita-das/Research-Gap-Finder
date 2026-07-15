"""
paper_loader.py

Loads paper information by combining:

1. Phase 1 merged metadata
2. Phase 2 extracted entities

Returns a temporary Paper Summary until the actual
Paper Reader Agent is implemented.
"""

from __future__ import annotations

import json
from pathlib import Path


class PaperLoader:
    """
    Loads paper metadata and extracted entities.

    Optimization:
    - merged.json is loaded once
    - all entity files are loaded once
    - subsequent lookups are dictionary lookups
    """

    def __init__(self, merged_file: str, entity_folder: str):

        self.merged_file = Path(merged_file)
        self.entity_folder = Path(entity_folder)

        # -------------------------------------------------
        # Load merged metadata once
        # -------------------------------------------------

        with open(self.merged_file, "r", encoding="utf-8") as f:
            merged_data = json.load(f)

        self.paper_lookup = {
            paper["paper_id"]: paper
            for paper in merged_data
        }

        # -------------------------------------------------
        # Load every entity file once
        # -------------------------------------------------

        self.entity_lookup = {}

        if self.entity_folder.exists():

            for entity_file in self.entity_folder.glob("*.json"):

                try:

                    with open(entity_file, "r", encoding="utf-8") as f:
                        entity = json.load(f)

                    paper_id = entity.get("paper_id")

                    if paper_id:
                        self.entity_lookup[paper_id] = entity

                except Exception as e:
                    print(f"Could not load {entity_file.name}: {e}")

        print(f"[PaperLoader] Loaded {len(self.paper_lookup)} papers.")
        print(f"[PaperLoader] Loaded {len(self.entity_lookup)} entity files.")

    # -------------------------------------------------

    def load(self, paper_id: str) -> dict:
        """
        Load one paper.

        Accepts either

            paper:arxiv_xxxxx

        or

            arxiv_xxxxx
        """

        if paper_id.startswith("paper:"):
            paper_id = paper_id.replace("paper:", "")

        metadata = self.paper_lookup.get(paper_id)

        if metadata is None:
            raise ValueError(f"Paper '{paper_id}' not found.")

        entities = self.entity_lookup.get(paper_id, {})

        paper_summary = {

            # --------------------------
            # Metadata
            # --------------------------

            "paper_id": metadata.get("paper_id", ""),
            "title": metadata.get("title", ""),
            "year": metadata.get("year", 0),
            "authors": metadata.get("authors", []),
            "abstract": metadata.get("abstract", ""),

            # --------------------------
            # Phase 2 entities
            # --------------------------

            "methods": entities.get("methods", []),
            "models": entities.get("models", []),
            "algorithms": entities.get("algorithms", []),
            "datasets": entities.get("datasets", []),
            "tasks": entities.get("tasks", []),
            "metrics": entities.get("metrics", []),
            "claims": entities.get("claims", []),
            "keywords": entities.get("keywords", []),
            "summary": entities.get("summary", ""),

            # --------------------------
            # Phase 6 enrichment
            # Placeholder values
            # --------------------------

            "experimental_results": [],
            "limitations": [],
            "future_work": [],
            "key_contributions": [],
            "novelty_points": [],
            "experimental_setup": "",
            "implementation_details": []
        }

        return paper_summary

    # -------------------------------------------------

    def load_multiple(self, paper_ids):
        """
        Load multiple papers.
        """

        return [self.load(pid) for pid in paper_ids]
            
    # -------------------------------------------------

    def load_all(self):
        """
        Load all papers as temporary Paper Summary objects.

        Returns:
            List[dict]
        """

        return [
            self.load(paper_id)
            for paper_id in self.paper_lookup
        ]


# -----------------------------------------------------
# Example usage
# -----------------------------------------------------

if __name__ == "__main__":

    loader = PaperLoader(
        merged_file="data/processed/merged/merged_RAG.json",
        entity_folder="data/processed/entities"
    )

    paper = loader.load("paper:arxiv_2409.03708v2")

    print(paper)