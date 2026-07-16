"""
prompts.py

Prompt templates for Phase 7.
"""

from __future__ import annotations

from src.research_idea_generation.schemas import HypothesisIdea
from src.research_idea_generation.gap_loader import MergedGap


def build_novelty_prompt(
    idea: HypothesisIdea,
    gap: MergedGap,
) -> str:

    return f"""
You are an expert research evaluator.

A validated research gap has already been identified.

The following research idea was generated from that gap.

--------------------------------------------------
Validated Gap
--------------------------------------------------

Title:
{gap.title}

Description:
{gap.description}

Supporting Evidence:
{gap.supporting_evidence}

Temporal Trend:
{gap.temporal_evidence.get("trend_summary","")}

Trend Type:
{gap.temporal_evidence.get("trend","")}

--------------------------------------------------
Generated Research Idea
--------------------------------------------------

Research Question:
{idea.research_question}

Hypothesis:
{idea.hypothesis}

Methodology:
{idea.proposed_methodology}

Expected Contribution:
{idea.expected_contribution}

Research Direction:
{idea.research_direction}

Graph Confidence:
{idea.graph_confidence:.4f}

Gap Confidence:
{idea.gap_confidence:.4f}

--------------------------------------------------

Evaluate this research idea.

Produce four scores.

1. Competition Score
How crowded does this research direction appear?

2. Impact Score
If successful, how impactful could this work be?

3. Novelty Score
How original does the proposed idea appear?

4. Feasibility Score
How feasible is this research using existing methods?

Rules

Rules

• Use ONLY the supplied information.
• Do NOT invent papers, datasets, benchmarks, evaluation metrics, or prior work.
• The publication counts shown in the temporal summary come only from the retrieved research corpus and should NOT be interpreted as global publication statistics.
• Use Graph Confidence and Gap Confidence only as supporting context about the reliability of the identified research gap. Do NOT directly convert them into scores.
• Every score must be between 0 and 1.
• Higher Competition Score means the area is more crowded.
• Higher Impact Score means the idea could make a larger contribution if successful.
• Higher Novelty Score means the idea appears more original.
• Higher Feasibility Score means the research appears more practical to execute.
• Provide one concise reason (1-2 sentences) for each score.

Return ONLY JSON.

{{
    "competition_score":0.0,
    "impact_score":0.0,
    "novelty_score":0.0,
    "feasibility_score":0.0,

    "competition_reason":"",
    "impact_reason":"",
    "novelty_reason":"",
    "feasibility_reason":""
}}
"""

def build_batch_novelty_prompt(
    ideas: list[HypothesisIdea],
    gaps: list[MergedGap],
) -> str:

    sections = []

    for idea, gap in zip(ideas, gaps):

        sections.append(f"""
==================================================
Gap ID
==================================================

{idea.gap_id}

Validated Gap

Title:
{gap.title}

Description:
{gap.description}

Supporting Evidence:
{gap.supporting_evidence}

Temporal Trend:
{gap.temporal_evidence.get("trend_summary", "")}

Trend Type:
{gap.temporal_evidence.get("trend", "")}

--------------------------------------------------

Generated Research Idea

Research Question:
{idea.research_question}

Hypothesis:
{idea.hypothesis}

Methodology:
{idea.proposed_methodology}

Expected Contribution:
{idea.expected_contribution}

Research Direction:
{idea.research_direction}

Graph Confidence:
{idea.graph_confidence:.4f}

Gap Confidence:
{idea.gap_confidence:.4f}
""")

    all_items = "\n".join(sections)

    return f"""
You are an expert research evaluator.

A set of validated research gaps and generated research ideas are provided.

Your task is to evaluate EACH idea independently.

==================================================

{all_items}

==================================================

For EACH gap return:

- competition_score
- impact_score
- novelty_score
- feasibility_score

and one concise reason for each score.

Rules

• Use ONLY the supplied information.
• Do NOT invent papers.
• Do NOT invent datasets.
• Do NOT invent benchmarks.
• Do NOT invent evaluation metrics.
• Do NOT invent prior work.
• Scores must be between 0 and 1.
• Higher competition score = more crowded research area.
• Higher novelty score = more original.
• Higher feasibility score = easier to execute.
• Higher impact score = larger expected contribution.

Return ONLY a JSON array.

Example

[
    {{
        "gap_id":"gap_0001",
        "competition_score":0.42,
        "impact_score":0.80,
        "novelty_score":0.73,
        "feasibility_score":0.67,

        "competition_reason":"",
        "impact_reason":"",
        "novelty_reason":"",
        "feasibility_reason":""
    }},
    {{
        "gap_id":"gap_0002",
        ...
    }}
]
"""