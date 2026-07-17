"""
comparison_agent.py

Google ADK Comparison Agent.
"""

from __future__ import annotations

from google.adk.agents import Agent

from Gap_finding_adk.schemas import ComparisonSummary

from Gap_finding_adk.tools.comparison_tool import (
    prepare_comparison,
)


COMPARISON_INSTRUCTION = """
You are an expert research paper comparison agent.

Your task is to compare two enriched research papers.

The comparison tool has already identified:

- shared methods
- shared datasets
- shared metrics
- different methods
- different datasets
- different metrics

Using ONLY the supplied information, identify:

- shared limitations
- complementary strengths
- missing connections
- an overall comparison summary

Rules

- Do NOT invent information.
- Use only the provided paper summaries.
- If evidence is insufficient,
  return empty lists.
- Keep the comparison concise.

Return a valid ComparisonSummary.
"""


comparison_agent = Agent(

    name="comparison_agent",

    description=(
        "Compares two enriched papers and identifies "
        "their similarities and complementary strengths."
    ),

    model="gemini-2.5-flash",

    instruction=COMPARISON_INSTRUCTION,

    tools=[
        prepare_comparison,
    ],

    output_schema=ComparisonSummary,

    output_key="comparison_summary",
)