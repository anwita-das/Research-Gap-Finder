"""
Google ADK Section Extraction Agent.
"""

from __future__ import annotations

from google.adk.agents import Agent

from ..tools.section_extractor_tool import (
    extract_sections,
)


SECTION_EXTRACTION_INSTRUCTION = """

You are an expert research paper analysis agent.

Your task is to extract structured scientific information
from a research paper.

Use the section extraction tool to obtain paper content.

Extract only factual scientific information.

Ignore:

- section titles
- table of contents
- navigation text
- incomplete sentences
- paper organization paragraphs

Do NOT invent information.

Return structured JSON matching:

{
    "methodology": [],
    "experimental_setup": [],
    "experimental_results": [],
    "discussion": [],
    "limitations": [],
    "future_work": [],
    "conclusion": []
}

"""


section_extractor_agent = Agent(

    name="section_extractor_agent",

    description=(
        "Extracts scientific sections "
        "from research papers."
    ),

    model="gemini-2.5-flash",

    instruction=SECTION_EXTRACTION_INSTRUCTION,

    tools=[
        extract_sections
    ],

    output_key="paper_sections",
)