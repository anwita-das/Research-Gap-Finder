from __future__ import annotations

import json

from src.Extraction.groq_client import GroqClient

from .schemas import PaperSections


class SectionExtractionError(Exception):
    """Raised when section extraction fails."""
    pass


class SectionExtractor:
    """
    Reads an entire paper by processing it in chunks.

    Each chunk is sent independently to the LLM.

    The extracted sections are merged into one
    PaperSections object.
    """

    def __init__(
        self,
        client: GroqClient | None = None,
        chunk_size: int = 3500,
    ):

        self.client = client or GroqClient()

        self.chunk_size = chunk_size

    # ---------------------------------------------------------

    def _chunk_text(
        self,
        paper_text: str,
    ) -> list[str]:
        """
        Split paper into overlapping chunks.

        Overlap prevents important sentences from being cut
        across chunk boundaries.
        """

        overlap = 500

        chunks = []

        start = 0

        while start < len(paper_text):

            end = min(
                start + self.chunk_size,
                len(paper_text),
            )

            chunks.append(
                paper_text[start:end]
            )

            if end == len(paper_text):
                break

            start = end - overlap

        return chunks

    # ---------------------------------------------------------

    def _build_prompt(
        self,
        chunk: str,
    ) -> str:

        return f"""
    You are reading ONE CHUNK of a research paper.

    Your job is NOT to summarize the chunk.

    Instead, extract EVERY piece of information related to the categories below.

    The information may appear anywhere in the chunk.
    Do NOT rely on section headings.

    For every category, return a LIST of short factual statements.

    If nothing is found for a category, return an empty list.

    IMPORTANT RULES

    - Do NOT extract table of contents entries.
    - Do NOT extract section titles.
    - Do NOT extract text like:
        - "Section 2 presents..."
        - "Section 5 describes..."
        - "The remainder of this paper..."
    - Ignore navigation text and paper organization paragraphs.
    - Extract only factual scientific content.
    - Ignore incomplete sentences.

    Return ONLY valid JSON.

    {{
        "methodology": [],
        "experimental_setup": [],
        "experimental_results": [],
        "discussion": [],
        "limitations": [],
        "future_work": [],
        "conclusion": []
    }}

    PAPER CHUNK

    {chunk}
    """

    # ---------------------------------------------------------

    def _merge(
        self,
        merged: dict[str, list[str]],
        extracted: dict,
    ):

        for key in merged:

            values = extracted.get(key, [])

            if not isinstance(values, list):
                continue

            for value in values:

                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    merged[key].append(value.strip())

    # ---------------------------------------------------------

    def extract_sections(
        self,
        paper_text: str,
    ) -> PaperSections:

        chunks = self._chunk_text(
            paper_text
        )

        merged = {
            "methodology": [],
            "experimental_setup": [],
            "experimental_results": [],
            "discussion": [],
            "limitations": [],
            "future_work": [],
            "conclusion": [],
        }

        try:

            for chunk in chunks:

                prompt = self._build_prompt(
                    chunk
                )

                response = self.client.generate(
                    prompt
                )

                extracted = json.loads(
                    response
                )

                self._merge(
                    merged,
                    extracted,
                )

        except Exception as exc:

            raise SectionExtractionError(
                "Failed to extract sections."
            ) from exc

        def unique(values):

            seen = set()

            output = []

            for value in values:

                key = value.lower().strip()

                if key in seen:
                    continue

                seen.add(key)

                output.append(value)

            return output

        return PaperSections(

            methodology=unique(
                merged["methodology"]
            ),

            experimental_setup=unique(
                merged["experimental_setup"]
            ),

            experimental_results=unique(
                merged["experimental_results"]
            ),

            discussion=unique(
                merged["discussion"]
            ),

            limitations=unique(
                merged["limitations"]
            ),

            future_work=unique(
                merged["future_work"]
            ),

            conclusion=unique(
                merged["conclusion"]
            ),
        )
    
if __name__ == "__main__":

    from dataclasses import asdict
    import json

    from .context_retriever import ContextRetriever
    from .pdf_reader import PDFReader

    retriever = ContextRetriever()

    candidate = retriever.load_candidates()[0]

    context = retriever.build_context(candidate)

    pdf_reader = PDFReader()

    paper_text = pdf_reader.extract_text(
        context.source_paper
    )

    extractor = SectionExtractor()

    sections = extractor.extract_sections(
        paper_text
    )

    print(
        json.dumps(
            asdict(sections),
            indent=2,
            ensure_ascii=False,
        )
    )