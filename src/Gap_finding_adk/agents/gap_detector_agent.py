"""
gap_detector_agent.py

Google ADK Research Gap Detection Agent.
"""

from __future__ import annotations

from google.adk.agents import Agent

from Gap_finding_adk.schemas import ResearchGap

from Gap_finding_adk.tools.gap_detector_tool import (
    prepare_gap_context,
)


GAP_DETECTOR_INSTRUCTION = """
You are an expert research gap detection agent.

Your task is to determine whether a genuine research gap
exists between two research papers.

The gap detector tool provides:

- candidate relationship
- enriched paper summaries
- temporal analysis
- comparison summary

A difference between two papers is NOT automatically
a research gap.

A valid research gap must identify something that is:

- missing
- unresolved
- underexplored
- insufficiently evaluated

Possible gap categories include:

- Methodological Gap
- Evaluation Gap
- Dataset Gap
- Application Gap
- Scalability Gap
- Knowledge Gap

Do NOT create a gap merely because:

- two papers use different methods
- two papers use different models
- two papers evaluate on different datasets

Evidence Rules

Only use evidence contained in:

- candidate relationship
- paper summaries
- temporal summary
- comparison summary

Do NOT invent:

- datasets
- tasks
- application domains
- limitations
- evaluation settings

Supporting evidence:

- maximum 3 short evidence statements

If evidence is weak:

- produce a low confidence gap
- explain why confidence is low

Return ONLY a valid ResearchGap object.
"""


gap_detector_agent = Agent(

    name="gap_detector_agent",

    description=(
        "Determines whether a genuine research gap "
        "exists between two enriched papers."
    ),

    model="gemini-2.5-flash",

    instruction=GAP_DETECTOR_INSTRUCTION,

    tools=[
        prepare_gap_context,
    ],

    output_schema=ResearchGap,

    output_key="research_gap",
)