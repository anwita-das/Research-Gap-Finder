"""
paper_loader.py

Loads enriched paper summaries produced by the
Paper Enrichment Pipeline.

Each paper is loaded once into memory and subsequent
lookups are dictionary lookups.
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

    def __init__(self, enriched_folder: str):

        self.enriched_folder = Path(enriched_folder)

        # -------------------------------------------------
        # Load enriched papers
        # -------------------------------------------------

        self.paper_lookup = {}

        if self.enriched_folder.exists():

            for paper_file in self.enriched_folder.glob("*.json"):

                try:

                    with open(
                        paper_file,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        paper = json.load(f)

                    paper_id = paper.get("paper_id")

                    if paper_id:

                        self.paper_lookup[paper_id] = paper

                except Exception as e:

                    print(
                        f"Could not load {paper_file.name}: {e}"
                    )

        print(
            f"[PaperLoader] Loaded {len(self.paper_lookup)} enriched papers."
        )

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

        paper = self.paper_lookup.get(paper_id)

        if paper is None:
            raise ValueError(
                f"Paper '{paper_id}' not found."
            )

        return paper

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
        enriched_folder="data/processed/enriched_papers"
    )

    paper = loader.load("paper:arxiv_2409.03708v2")

    print(paper)