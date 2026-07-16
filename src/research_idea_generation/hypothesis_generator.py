"""
hypothesis_generator.py

Phase 7 - Step 1

Hypothesis Generator Agent

"""

from __future__ import annotations

import json
from pathlib import Path

from src.gap_finding.llm_helper import LLMHelper
from dataclasses import asdict

from src.research_idea_generation.gap_loader import MergedGap
from src.research_idea_generation.schemas import HypothesisIdea


class HypothesisGenerator:

    def __init__(self):

        self.llm = LLMHelper()

    # =====================================================
    # MAIN FUNCTION
    # =====================================================

    def generate(
        self,
        gaps: list[MergedGap],
    ) -> list[HypothesisIdea]:

        cached = self.load_cache()

        if cached is not None:
            print("Loading hypothesis ideas from cache")
            return cached

        prompt = self._build_prompt(gaps)

        try:

            response = self.llm.generate(prompt)

            response = response.strip()

            if response.startswith("```"):

                response = (
                    response
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            result = json.loads(response)

        except Exception as e:

            print("Hypothesis generation failed:", e)

            return []

        gap_lookup = {

            gap.gap_id: gap

            for gap in gaps

        }

        ideas = []

        print(
            f"Generated {len(result)} ideas for {len(gaps)} gaps"
        )

        for item in result:

            gap_id = item.get("gap_id")

            if gap_id not in gap_lookup:
                print(
                    f"Unknown gap id returned: {gap_id}"
                )
                continue

            gap = gap_lookup[gap_id]

            ideas.append(

                HypothesisIdea(

                    gap_id=gap.gap_id,

                    graph_confidence=gap.graph_confidence,

                    gap_confidence=gap.gap_confidence,

                    research_question=item.get(
                        "research_question",
                        "",
                    ),

                    hypothesis=item.get(
                        "hypothesis",
                        "",
                    ),

                    proposed_methodology=item.get(
                        "proposed_methodology",
                        "",
                    ),

                    supporting_explanation=item.get(
                        "supporting_explanation",
                        "",
                    ),

                    expected_contribution=item.get(
                        "expected_contribution",
                        "",
                    ),

                    research_direction=item.get(
                        "research_direction",
                        "",
                    ),

                )

            )
        output_path = Path(
            "data/processed/hypothesis_ideas.json"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                [asdict(idea) for idea in ideas],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved {len(ideas)} ideas to {output_path}")

        return ideas

    # =====================================================
    # PROMPT
    # =====================================================

    def _build_prompt(
        self,
        gaps: list[MergedGap]
    ) -> str:

        gap_blocks = []

        for gap in gaps:

            gap_blocks.append(

                f"""
    ========================================================
    Gap ID:
    {gap.gap_id}

    Gap Type:
    {gap.gap_type}

    Title:
    {gap.title}

    Description:
    {gap.description}

    Supporting Evidence:
    {json.dumps(gap.supporting_evidence, indent=2)}

    Comparison Summary:
    {gap.comparison_summary}

    Temporal Summary:
    {gap.temporal_evidence.get("trend_summary", "")}

    Research Trend:
    {gap.temporal_evidence.get("trend", "")}

    Shared Research Entity:
    {gap.shared_entity}
    """
            )

        all_gaps = "\n".join(gap_blocks)
        print(f"Generating ideas for {len(gaps)} gaps...")

        return f"""
    You are an expert AI research advisor.

    A set of validated research gaps has already been identified.

    Your task is NOT to detect new gaps.

    Instead, generate ONE concrete research idea for EACH validated research gap.

    ========================================================
    Validated Research Gaps
    ========================================================

    {all_gaps}

    ========================================================

    For EACH gap, generate exactly one research idea.

    Rules:

    1. The research question must directly address the validated research gap.

    2. Do NOT introduce new application domains unless they appear in the validated gap.

    3. Do NOT invent datasets.

    4. Do NOT invent evaluation metrics.

    5. Do NOT invent benchmark tasks.

    6. Do NOT invent baseline models.

    7. Base every statement ONLY on:
    - validated gap description
    - supporting evidence
    - comparison summary
    - temporal evidence

    8. The hypothesis must be experimentally testable.

    9. The methodology should describe only a high-level research plan.

    10. If datasets, benchmarks or architectures are not explicitly mentioned,
        refer only to:
            - existing methods
            - existing evaluation settings
            - existing research approaches

    11. The expected contribution should clearly explain how the proposed work fills the validated gap.

    12. The research direction should contain 3-6 words.

    13. If the validated gap mentions limitations or future work,
        the generated idea should directly address them.

    14. Return ONE idea for EVERY gap.

    Return ONLY a JSON array.

    Example:

    [
        {{
            "gap_id":"gap_0001",
            "research_question":"",
            "hypothesis":"",
            "proposed_methodology":"",
            "supporting_explanation":"",
            "expected_contribution":"",
            "research_direction":""
        }},
        {{
            "gap_id":"gap_0002",
            "research_question":"",
            "hypothesis":"",
            "proposed_methodology":"",
            "supporting_explanation":"",
            "expected_contribution":"",
            "research_direction":""
        }}
    ]

    Return ONLY JSON.
    """

    def load_cache(self):

        path = Path(
            "data/processed/hypothesis_ideas.json"
        )

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:

            data = json.load(f)

        return [
            HypothesisIdea(**item)
            for item in data
        ]

if __name__ == "__main__":

    from src.research_idea_generation.gap_loader import GapLoader

    loader = GapLoader(

        candidate_file="data/processed/final_gap_candidates.json",

        gap_file="data/processed/research_gaps.json"

    )

    gaps = loader.load()

    generator = HypothesisGenerator()

    idea = generator.generate(gaps)

    print()

    print(idea)