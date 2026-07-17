"""
Section Extraction Tool

Google ADK tool responsible for preparing
research paper content for the Section Extraction Agent.

Responsibilities:
- Read paper PDF
- Extract text
- Split text into chunks

It does NOT call an LLM.
The ADK agent performs extraction reasoning.
"""

from __future__ import annotations

from typing import Any

from gap_finding.pdf_reader import PDFReader

from ..section_processing import chunk_text


class SectionExtractionTool:
    """
    Tool used by section_extractor_agent.

    Provides paper chunks that the agent
    analyzes and converts into structured sections.
    """

    def __init__(
        self,
        chunk_size: int = 8000,
        overlap: int = 500,
    ):

        self.reader = PDFReader()

        self.chunk_size = chunk_size

        self.overlap = overlap


    def prepare_paper(
        self,
        paper: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract paper text and split into chunks.

        Parameters
        ----------
        paper:
            Paper metadata containing pdf_url,
            paper_id, etc.

        Returns
        -------
        dict:
            {
                "paper_id": "...",
                "chunks": [...]
            }
        """

        text = self.reader.extract_text(
            paper
        )


        if not text:

            return {
                "paper_id": paper.get(
                    "paper_id",
                    "unknown"
                ),
                "chunks": [],
            }


        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )


        return {

            "paper_id": paper.get(
                "paper_id",
                "unknown"
            ),

            "chunks": chunks,

        }



# ---------------------------------------------------------
# ADK tool instance
# ---------------------------------------------------------

section_extractor_tool = SectionExtractionTool()


# ---------------------------------------------------------
# Tool function exposed to ADK agent
# ---------------------------------------------------------

def extract_sections(
    paper: dict[str, Any],
) -> dict[str, Any]:
    """
    ADK callable tool.

    The section extractor agent calls this function.
    """

    return section_extractor_tool.prepare_paper(
        paper
    )