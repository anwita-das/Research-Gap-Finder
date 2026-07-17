"""
enrichment_agent.py

Google ADK Paper Enrichment Agent.
"""

from __future__ import annotations

from google.adk.agents import Agent


ENRICHMENT_INSTRUCTION = """
You are an expert research paper analysis agent.

Your task is to enrich a structured representation of a research paper.

You will be given:

- Paper metadata
- Existing entity extraction
- Extracted paper sections

Using ONLY the supplied information, produce an enriched paper summary.

Rules

- Do NOT invent information.
- Do NOT infer datasets, models, or metrics that are not explicitly stated.
- Do NOT hallucinate experimental results.
- Preserve existing extracted entities.
- Summarize only factual information supported by the paper.

Return ONLY valid JSON matching the required PaperSummary schema.
"""


enrichment_agent = Agent(

    name="enrichment_agent",

    description=(
        "Enriches a research paper into a structured "
        "PaperSummary representation."
    ),

    model="gemini-2.5-flash",

    instruction=ENRICHMENT_INSTRUCTION,

    output_key="paper_summary",

)