from __future__ import annotations

import re
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

    def __init__(self):
        pass
    # ---------------------------------------------------------

    def _extract_section(self, text: str, headings: list[str]) -> list[str]:

        """
            Extract the text belonging to one section using regex headings.
            """

        pattern = (
                r"(?is)"
                r"(?:^|\n)\s*"
                + "(" + "|".join(re.escape(h) for h in headings) + ")"
                + r"\s*\n"
                + r"(.*?)"
                + r"(?=\n\s*[A-Z][A-Za-z0-9 \-]{1,60}\n|\Z)"
            )

        matches = re.findall(pattern, text)

        return [
                body.strip()
                for _, body in matches
                if body.strip()
            ]


    # ---------------------------------------------------------

    def extract_sections(
        self,
        paper_text: str,
    ) -> PaperSections:

        methodology = self._extract_section(
            paper_text,
            [
                "Method",
                "Methods",
                "Methodology",
                "Approach",
                "Proposed Method",
                "Framework",
            ],
        )

        experimental_setup = self._extract_section(
            paper_text,
            [
                "Experimental Setup",
                "Experiments",
                "Evaluation Setup",
                "Implementation Details",
            ],
        )

        experimental_results = self._extract_section(
            paper_text,
            [
                "Results",
                "Experimental Results",
                "Evaluation",
            ],
        )

        discussion = self._extract_section(
            paper_text,
            [
                "Discussion",
            ],
        )

        limitations = self._extract_section(
            paper_text,
            [
                "Limitations",
            ],
        )

        future_work = self._extract_section(
            paper_text,
            [
                "Future Work",
                "Future Directions",
            ],
        )

        conclusion = self._extract_section(
            paper_text,
            [
                "Conclusion",
                "Conclusions",
            ],
        )

        return PaperSections(
            methodology=methodology,
            experimental_setup=experimental_setup,
            experimental_results=experimental_results,
            discussion=discussion,
            limitations=limitations,
            future_work=future_work,
            conclusion=conclusion,
        )

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