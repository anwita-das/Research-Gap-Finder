"""
temporal_agent.py

Google ADK Temporal Analysis Agent.
"""

from __future__ import annotations

from google.adk.agents import Agent

from Gap_finding_adk.schemas import TemporalSummary

from Gap_finding_adk.tools.temporal_tool import (
    prepare_temporal_data,
)


TEMPORAL_INSTRUCTION = """
You are an expert research trend analyst.

Your task is to analyze the temporal evolution of a
research topic.

Use the temporal statistics provided by the tool.

Describe:

- publication evolution
- maturity of the topic
- whether the topic is:
    - Emerging
    - Stable
    - Declining

Rules

- Use ONLY the supplied statistics.
- Do NOT invent publication years.
- Do NOT invent citation trends.
- Keep the analysis concise.
"""


temporal_agent = Agent(

    name="temporal_agent",

    description=(
        "Analyzes publication timelines and "
        "research evolution."
    ),

    model="gemini-2.5-flash",

    instruction=TEMPORAL_INSTRUCTION,

    tools=[
        prepare_temporal_data,
    ],

    output_schema=TemporalSummary,

    output_key="temporal_summary",
)