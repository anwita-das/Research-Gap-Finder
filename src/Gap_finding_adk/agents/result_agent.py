"""
result_agent.py

Final ADK agent.

Packages the final research gap result produced by the
Gap Detector Agent.
"""

from google.adk.agents import Agent

RESULT_INSTRUCTION = """
You are the final result agent.

Your job is to present the final research gap detected by the previous agents.

Do NOT invent information.

Return only the final structured research gap containing:

- gap_id
- gap_type
- title
- description
- supporting_evidence
- confidence
- comparison_summary
- temporal_evidence

Do not modify any values.
"""

result_agent = Agent(
    name="result_agent",
    description="Returns the final research gap result.",
    model="gemini-2.5-flash",
    instruction=RESULT_INSTRUCTION,
    output_key="final_gap",
)