"""
comparison_agent.py

Phase 6 - Comparison Agent

Responsibilities:
- Compare enriched paper summaries
- Identify:
    - Shared methods
    - Different methods
    - Different datasets
    - Different metrics
    - Complementary strengths
    - Shared limitations
    - Missing connections

Output:
Comparison Summary
"""


from __future__ import annotations

import json
from typing import List

from src.gap_finding.llm_helper import LLMHelper



class ComparisonAgent:


    def __init__(self):

        self.llm = LLMHelper()



    # ==================================================
    # MAIN FUNCTION
    # ==================================================

    def compare(
        self,
        candidate: dict,
        paper1: dict,
        paper2: dict,
        temporal_summary: dict | None = None
    ) -> dict:


        candidate_id = candidate["gap_id"]



        # ----------------------------------------------
        # Entity comparison
        # ----------------------------------------------

        shared_methods = self._intersection(
            paper1.get("methods", []),
            paper2.get("methods", [])
        )


        shared_datasets = self._intersection(
            paper1.get("datasets", []),
            paper2.get("datasets", [])
        )


        shared_metrics = self._intersection(
            paper1.get("metrics", []),
            paper2.get("metrics", [])
        )



        different_methods = (
            self._difference(
                paper1.get("methods", []),
                paper2.get("methods", [])
            )
            +
            self._difference(
                paper2.get("methods", []),
                paper1.get("methods", [])
            )
        )



        different_datasets = (
            self._difference(
                paper1.get("datasets", []),
                paper2.get("datasets", [])
            )
            +
            self._difference(
                paper2.get("datasets", []),
                paper1.get("datasets", [])
            )
        )



        different_metrics = (
            self._difference(
                paper1.get("metrics", []),
                paper2.get("metrics", [])
            )
            +
            self._difference(
                paper2.get("metrics", []),
                paper1.get("metrics", [])
            )
        )



        # ----------------------------------------------
        # LLM reasoning
        # ----------------------------------------------

        llm_result = self._generate_comparison(

            paper1,
            paper2,

            shared_methods,
            shared_datasets,
            shared_metrics

        )



        return {

            "candidate_id": candidate_id,


            "shared_methods": shared_methods,


            "shared_datasets": shared_datasets,


            "different_methods": different_methods,


            "different_datasets": different_datasets,


            "different_metrics": different_metrics,


            "shared_limitations":
                llm_result.get(
                    "shared_limitations",
                    []
                ),


            "complementary_strengths":
                llm_result.get(
                    "complementary_strengths",
                    []
                ),


            "missing_connections":
                llm_result.get(
                    "missing_connections",
                    []
                ),


            "comparison_summary":
                llm_result.get(
                    "comparison_summary",
                    ""
                )
        }



    # ==================================================
    # NORMALIZATION
    # ==================================================

    def _normalize_entity(
        self,
        text: str
    ) -> str:

        return (
            text
            .lower()
            .replace("-", " ")
            .replace("_", " ")
            .strip()
        )



    # ==================================================
    # LIST OPERATIONS
    # ==================================================

    def _intersection(
        self,
        list1: List,
        list2: List
    ):


        normalized1 = {

            self._normalize_entity(x): x

            for x in list1

        }


        normalized2 = {

            self._normalize_entity(x): x

            for x in list2

        }


        return [

            normalized1[key]

            for key in normalized1

            if key in normalized2

        ]



    def _difference(
        self,
        list1: List,
        list2: List
    ):


        normalized2 = {

            self._normalize_entity(x)

            for x in list2

        }


        return [

            x

            for x in list1

            if self._normalize_entity(x)
            not in normalized2

        ]



    # ==================================================
    # LLM GENERATION
    # ==================================================

    def _generate_comparison(
        self,
        paper1,
        paper2,
        shared_methods,
        shared_datasets,
        shared_metrics
    ):


        prompt = f"""

You are a research paper comparison analyst.

Compare these two papers.


Paper 1:

Title:
{paper1.get("title")}

Methods:
{paper1.get("methods")}

Datasets:
{paper1.get("datasets")}

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

Experimental Results:
{paper1.get("experimental_results")}

Paper 2:

Title:
{paper2.get("title")}

Methods:
{paper2.get("methods")}

Datasets:
{paper2.get("datasets")}

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

Experimental Results:
{paper2.get("experimental_results")}

Existing overlap:

Shared methods:
{shared_methods}

Shared datasets:
{shared_datasets}

Shared metrics:
{shared_metrics}



Return ONLY JSON:

{{
    "shared_limitations": [],
    "complementary_strengths": [],
    "missing_connections": [],
    "comparison_summary": ""
}}



Rules:

- Do not invent information.
- Use only the provided paper information.
- If information is unavailable, return empty lists.
- Keep summary concise.

"""


        try:

            print(f"Prompt length: {len(prompt)} characters")

            response = self.llm.generate(prompt)

            print("\n" + "=" * 80)
            print("RAW COMPARISON RESPONSE")
            print("=" * 80)
            print(repr(response))
            print("=" * 80 + "\n")

            # Remove markdown code fences if present
            cleaned_response = response.strip()

            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```json", "")
                cleaned_response = cleaned_response.replace("```", "")
                cleaned_response = cleaned_response.strip()

            return json.loads(cleaned_response)

        except Exception as e:


            print("\nComparison Agent Error")
            print(type(e).__name__)
            print(e)

            try:
                print("\nRaw response was:")
                print(repr(response))
            except NameError:
                print("No response received from LLM.")

            print()


            return {

                "shared_limitations": [],

                "complementary_strengths": [],

                "missing_connections": [],

                "comparison_summary":
                    "Comparison generated without LLM enrichment."

            }