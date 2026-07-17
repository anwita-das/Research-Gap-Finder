"""
root_agent.py

Root Google ADK Agent.

Coordinates the entire Phase 6 workflow.
"""

from google.adk.agents import Agent

from .section_extractor_agent import section_extractor_agent
from .enrichment_agent import enrichment_agent
from .temporal_agent import temporal_agent
from .comparison_agent import comparison_agent
from .gap_detector_agent import gap_detector_agent
from .result_agent import result_agent


ROOT_INSTRUCTION = """
You coordinate the complete research gap detection workflow.

Execute the following workflow in order:

1. Section Extraction Agent
   - Extract structured scientific sections from both papers.

2. Enrichment Agent
   - Produce enriched PaperSummary objects.

3. Temporal Agent
   - Analyze temporal evolution between papers.

4. Comparison Agent
   - Compare the enriched papers.

5. Gap Detector Agent
   - Determine whether a genuine research gap exists.

6. Result Agent
   - Return the final structured research gap.

Never skip a step.

Pass the output of one agent to the next.

The Result Agent produces the final answer.
"""


root_agent = Agent(
    name="research_gap_root_agent",
    description="Coordinates the complete research gap detection workflow.",
    model="gemini-2.5-flash",
    instruction=ROOT_INSTRUCTION,
    sub_agents=[
        section_extractor_agent,
        enrichment_agent,
        temporal_agent,
        comparison_agent,
        gap_detector_agent,
        result_agent,
    ],
)