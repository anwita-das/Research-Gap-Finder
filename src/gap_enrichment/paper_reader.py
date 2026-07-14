from __future__ import annotations

from .schemas import (
    PaperSummary,
    PaperSections,
)

from .prompts import (
    build_enrichment_prompt,
)

from .llm_parser import (
    parse_enrichment_response,
)

from src.Extraction.groq_client import (
    GroqClient,
)

class PaperEnrichmentError(Exception):
    """Raised when a paper cannot be enriched."""
    pass

class PaperReader:
    """
    Uses an LLM to enrich a single paper.

    This class does NOT perform any filesystem operations.
    It simply converts

        Paper Metadata
        +
        Existing Entity Extraction

    into

        PaperSummary
    """

    def __init__(
        self,
        client: GroqClient | None = None
    ):

        self.client = client or GroqClient()

    def enrich_paper(
        self,
        paper: dict,
        entities: dict,
        sections: PaperSections,
    ) -> PaperSummary:
        """
        Enrich one paper.

        Parameters
        ----------
        paper
            Metadata loaded from merged_RAG.json

        entities
            Existing entity extraction from Phase 2

        Returns
        -------
        PaperSummary
        """

        try:

            prompt = build_enrichment_prompt(
                paper,
                entities,
                sections,
            )

            response = self.client.generate(
                prompt
            )

            summary = parse_enrichment_response(
                response,
                paper,
                entities
            )

            return summary

        except Exception as exc:

            raise PaperEnrichmentError(
                f"Failed to enrich paper "
                f"{paper.get('paper_id', 'Unknown')}"
            ) from exc
        
if __name__ == "__main__":

    from .context_retriever import ContextRetriever
    from dataclasses import asdict
    import json

    retriever = ContextRetriever()

    candidate = retriever.load_candidates()[0]

    context = retriever.build_context(candidate)

    from .pdf_reader import PDFReader

    pdf_reader = PDFReader()

    paper_text = pdf_reader.extract_text(
        context.source_paper
    )

    reader = PaperReader()

    from .section_extractor import SectionExtractor

    extractor = SectionExtractor()

    sections = extractor.extract_sections(
        paper_text
    )

    summary = reader.enrich_paper(
        context.source_paper,
        context.source_entities,
        sections,
    )

    print(
        json.dumps(
            asdict(summary),
            indent=2,
            ensure_ascii=False,
        )
    )