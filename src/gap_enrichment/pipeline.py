from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from .pdf_reader import PDFReader
from .section_extractor import SectionExtractor

from .context_retriever import ContextRetriever
from .paper_reader import (
    PaperReader,
    PaperEnrichmentError,
)


class GapEnrichmentPipeline:
    """
    Phase 6 - Paper Enrichment Pipeline.

    Reads candidate edges, enriches every unique paper exactly once,
    and stores the enriched summaries for downstream gap detection.
    """

    def __init__(self):

        self.retriever = ContextRetriever()

        self.reader = PaperReader()
        self.pdf_reader = PDFReader()
        self.section_extractor = SectionExtractor()

        self.output_dir = (
            self.retriever.data_dir /
            "enriched_papers"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==========================================================
    # Helper Methods
    # ==========================================================

    def _paper_filename(
        self,
        paper_id: str,
    ) -> Path:

        return self.output_dir / f"{paper_id}.json"

    def _is_cached(
        self,
        paper_id: str,
    ) -> bool:

        return self._paper_filename(
            paper_id
        ).exists()

    def _save_summary(
        self,
        summary,
    ):

        output_file = self._paper_filename(
            summary.paper_id
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                asdict(summary),
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _get_unique_papers(
        self,
    ) -> set[str]:
        """
        Collect every unique paper appearing
        in the candidate edges.
        """

        paper_ids = set()

        candidates = self.retriever.load_candidates()

        for candidate in candidates:

            paper_ids.add(
                candidate.source_node.replace(
                    "paper:",
                    "",
                    1,
                )
            )

            paper_ids.add(
                candidate.target_node.replace(
                    "paper:",
                    "",
                    1,
                )
            )

        return paper_ids

    # ==========================================================
    # Main Pipeline
    # ==========================================================

    def run(self):

        paper_ids = self._get_unique_papers()

        print(
            f"\nFound {len(paper_ids)} unique papers.\n"
        )

        enriched = 0
        skipped = 0
        failed = 0

        for paper_id in sorted(paper_ids):

            if self._is_cached(
                paper_id
            ):

                print(
                    f"[SKIP] {paper_id}"
                )

                skipped += 1

                continue

            paper = self.retriever.paper_map.get(
                paper_id
            )

            entities = self.retriever.entity_map.get(
                paper_id,
                {},
            )

            if paper is None:

                print(
                    f"[MISSING] {paper_id}"
                )

                failed += 1

                continue

            try:

                paper_text = self.pdf_reader.extract_text(
                    paper
                )

                sections = self.section_extractor.extract_sections(
                    paper_text
                )

                summary = self.reader.enrich_paper(
                    paper,
                    entities,
                    sections,
                )

                self._save_summary(
                    summary
                )

                print(
                    f"[DONE] {paper_id}"
                )

                enriched += 1

            except PaperEnrichmentError as exc:

                print(
                    f"[FAILED] {paper_id}"
                )

                print(exc)

                failed += 1

        print("\n==========================")
        print("Pipeline Complete")
        print("==========================")

        print(
            f"Enriched : {enriched}"
        )

        print(
            f"Skipped  : {skipped}"
        )

        print(
            f"Failed   : {failed}"
        )


def main():

    pipeline = GapEnrichmentPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()