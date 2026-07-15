"""
gap_detector.py

Phase 6 - Gap Detector Agent

Responsibilities:
- Combine:
    Candidate Edge
    Paper Summaries
    Temporal Summary
    Comparison Summary

- Decide whether a genuine research gap exists

Output:
Research Gap
"""


from __future__ import annotations

import json

from src.gap_finding.llm_helper import LLMHelper



class GapDetector:


    def __init__(self):

        self.llm = LLMHelper()



    # ==================================================
    # MAIN FUNCTION
    # ==================================================

    def detect_gap(
        self,
        candidate: dict,
        paper1: dict,
        paper2: dict,
        temporal_summary: dict,
        comparison_summary: dict
    ) -> dict:


        gap_id = candidate.get(
            "gap_id",
            ""
        )


        prompt = f"""

You are an expert research gap detection agent.

Your task is to determine whether a REAL research gap exists between
two research papers.

Important:

A difference between two papers is NOT automatically a research gap.

A valid research gap must identify something missing, unresolved,
underexplored, or insufficiently evaluated.


--------------------------------------------------
Candidate Relationship
--------------------------------------------------

Candidate ID:
{candidate.get("gap_id")}

Relationship:
{candidate.get("relation")}

Shared Entity:
{candidate.get("shared_entity")}

Graph Confidence:
{candidate.get("confidence")}


--------------------------------------------------
Paper 1
--------------------------------------------------

Title:
{paper1.get("title")}

#  Abstract:
#  {paper1.get("abstract")}

Methods:
{paper1.get("methods")}

Models:
{paper1.get("models")}

Tasks:
{paper1.get("tasks")}

Metrics:
{paper1.get("metrics")}

Limitations:
{paper1.get("limitations")}

Future Work:
{paper1.get("future_work")}
Key Contributions:
{paper1.get("key_contributions")}

Novelty Points:
{paper1.get("novelty_points")}

Experimental Setup:
{paper1.get("experimental_setup")}

Experimental Results:
{paper1.get("experimental_results")}



--------------------------------------------------
Paper 2
--------------------------------------------------

Title:
{paper2.get("title")}

# Abstract:
# {paper2.get("abstract")}

Methods:
{paper2.get("methods")}

Models:
{paper2.get("models")}

Tasks:
{paper2.get("tasks")}

Metrics:
{paper2.get("metrics")}

Limitations:
{paper2.get("limitations")}

Future Work:
{paper2.get("future_work")}
Key Contributions:
{paper2.get("key_contributions")}

Novelty Points:
{paper2.get("novelty_points")}

Experimental Setup:
{paper2.get("experimental_setup")}

Experimental Results:
{paper2.get("experimental_results")}




--------------------------------------------------
Temporal Evidence
--------------------------------------------------

{json.dumps(
    temporal_summary,
    indent=2
)}



--------------------------------------------------
Comparison Evidence
--------------------------------------------------

{json.dumps(
    comparison_summary,
    indent=2
)}



--------------------------------------------------
Gap Reasoning Rules
--------------------------------------------------

A research gap may be:

1. Methodological Gap
   - Existing methods fail to address an important limitation.
   - A missing combination of approaches exists.

2. Evaluation Gap
   - Methods are evaluated using inconsistent metrics,
     missing benchmarks, or incomplete comparisons.

3. Dataset Gap
   - Important datasets, domains, or scenarios are missing.

4. Application Gap
   - A technique has not been explored in an important domain.

5. Scalability Gap
   - Existing approaches do not address efficiency,
     large-scale deployment, or resource limitations.

6. Knowledge Gap
   - Important research questions remain unanswered.


Do NOT create a gap only because:
- two papers use different methods
- two papers use different architectures
- one paper has more metrics than another

Evidence Constraint:

Only generate claims that are directly supported by:

1. Paper metadata
2. Extracted entities
3. Paper summary enrichment fields
4. Comparison summary
5. Temporal summary


Supporting evidence rules:

- Generate maximum 3 evidence items.
- Each evidence item must be one short sentence.
- Evidence should summarize the information.
- Do not copy sentences from abstracts.
- Do not include detailed experimental descriptions unless present in paper summaries.

If evidence is insufficient:
- create a low confidence gap
- explain that evidence is weak

If the only difference between papers is their methods,
do not call it a research gap unless a limitation,
missing evaluation, or unexplored direction is identified.

Do not introduce:
- new application domains
- new datasets
- new tasks
- new limitations
- new evaluation scenarios

unless explicitly present.
Return ONLY JSON:

{{
    "gap_type": "",
    "title": "",
    "description": "",
    "supporting_evidence": [],
    "confidence": 0.0
}}

"""


        try:

            response = self.llm.generate(
                prompt
            )


            result = json.loads(
                response
            )


        except Exception as e:


            print(
                "LLM gap detection failed:",
                e
            )


            result = {

                "gap_type":
                    "unknown",

                "title":
                    "",

                "description":
                    "Gap detection failed.",

                "supporting_evidence":
                    [],

                "confidence":
                    0.0
            }



        return {

            "gap_id":
                gap_id,


            "candidate_id":
                gap_id,


            "gap_type":
                result.get(
                    "gap_type",
                    ""
                ),


            "title":
                result.get(
                    "title",
                    ""
                ),


            "description":
                result.get(
                    "description",
                    ""
                ),


            "supporting_evidence":
                result.get(
                    "supporting_evidence",
                    []
                ),


            "comparison_summary":
                comparison_summary.get(
                    "comparison_summary",
                    ""
                ),


            "temporal_evidence":
                temporal_summary,


            "confidence":
                result.get(
                    "confidence",
                    0.0
                )

        }