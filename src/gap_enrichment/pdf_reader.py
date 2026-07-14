from __future__ import annotations

import fitz  # PyMuPDF
import requests
from pathlib import Path
import re

class PDFReader:
    """
    Downloads and extracts text from research papers.
    """

    def __init__(self):

        self.cache_dir = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "pdf_cache"
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _download_pdf(
        self,
        paper_id: str,
        pdf_url: str,
    ) -> Path:

        pdf_path = self.cache_dir / f"{paper_id}.pdf"

        if pdf_path.exists():
            return pdf_path

        response = requests.get(
            pdf_url,
            timeout=60,
        )

        response.raise_for_status()

        pdf_path.write_bytes(response.content)

        return pdf_path

    def extract_text(
        self,
        paper: dict,
    ) -> str:
        """
        Download the paper if necessary and extract text.
        """

        pdf_url = paper.get("pdf_url", "")

        if not pdf_url:
            return ""

        try:

            pdf_path = self._download_pdf(
                paper.get("paper_id", "unknown"),
                pdf_url,
            )

            document = fitz.open(pdf_path)

            text = []

            for page in document:
                text.append(page.get_text())

            document.close()

        except Exception:
            return ""

        paper_text = "\n".join(text)

        return self._extract_relevant_sections(paper_text)
    
    def _extract_relevant_sections(
        self,
        paper_text: str,
    ) -> str:
        """
        Extract only the sections useful for research-gap analysis.
        """

        headings = [

            "abstract",

            "introduction",

            "method",

            "methodology",

            "approach",

            "proposed method",

            "implementation",

            "experimental setup",

            "experiments",

            "evaluation",

            "results",

            "discussion",

            "limitations",

            "future work",

            "conclusion",

        ]

        lower = paper_text.lower()

        matches = []

        for heading in headings:

            for match in re.finditer(
                rf"\b{re.escape(heading)}\b",
                lower,
            ):

                matches.append(
                    (
                        match.start(),
                        heading,
                    )
                )

        if not matches:
            return paper_text[:12000]

        matches.sort()

        sections = []

        for i, (start, heading) in enumerate(matches):

            if i + 1 < len(matches):
                end = matches[i + 1][0]
            else:
                end = len(paper_text)

            section = paper_text[start:end].strip()

            if len(section) < 50:
                continue

            sections.append(
                f"\n\n===== {heading.upper()} =====\n\n"
                + section
            )

        combined = "\n".join(sections)

        MAX_SECTION_TEXT = 15000

        if len(combined) > MAX_SECTION_TEXT:
            combined = combined[:MAX_SECTION_TEXT]

        return combined

if __name__ == "__main__":

    from .context_retriever import ContextRetriever

    retriever = ContextRetriever()

    candidate = retriever.load_candidates()[0]

    context = retriever.build_context(candidate)

    reader = PDFReader()

    text = reader.extract_text(
        context.source_paper
    )

    print(text[:3000])

    print("\n")
    print("=" * 60)
    print(f"Characters extracted: {len(text)}")