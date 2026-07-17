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

    Your goal is to transform each validated research gap into a publishable research proposal.

    A publishable proposal should introduce a concrete technical idea rather than merely recommending additional evaluation or comparison of existing work.

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

    7. Every proposed research idea MUST explicitly address at least ONE concrete limitation, contradiction, or unresolved issue mentioned in the supporting evidence or comparison summary.

    Do NOT generate generic RAG improvement ideas that could apply to any paper.

    The proposed solution should clearly map to the specific validated gap.

    8. The hypothesis must describe WHY the proposed research approach is expected to solve the identified limitation. It should make a testable scientific claim rather than simply stating that the research should be conducted.

    9. The methodology must describe:

        • what existing component is being changed
        • what new mechanism is introduced
        • why that mechanism addresses the identified gap

        Avoid vague phrases such as:

        - investigate
        - explore
        - study
        - analyze
        - compare
        - evaluate

        unless the validated gap is specifically about benchmarking.

        The methodology should describe a concrete research design rather than a research objective.

    10. Prefer proposing a new research mechanism over proposing a new evaluation.

        Examples of acceptable mechanisms include:

        - adaptive retrieval strategy
        - dynamic retrieval selection
        - confidence-aware retrieval
        - hierarchical retrieval
        - iterative retrieval refinement
        - retrieval orchestration
        - retrieval planning
        - retrieval quality estimation
        - query decomposition
        - adaptive context compression

        Do NOT invent fictional datasets, benchmark suites, published models, or prior work.

        You MAY propose a novel algorithmic mechanism or architectural modification if it is logically derived from the validated research gap.

    11. The expected contribution should explain what new capability, methodology, framework, optimization strategy, or architectural improvement the work introduces.

    12. The research direction should contain 3-6 words.

    13. If the validated gap mentions limitations or future work,
        the generated idea should directly address them.

    14. Avoid beginning research questions with:

    - How effective...
    - Can existing...
    - Compare...
    - Evaluate...
    - Investigate...

    unless the validated gap is explicitly about benchmarking or evaluation.

    Instead, formulate questions that propose a new solution to the identified limitation.

    15. The hypothesis must predict the expected effect of the proposed mechanism.

    A good hypothesis has the form:

    "Introducing X will improve Y because Z."

    Avoid hypotheses that merely restate the research question.

    16. A good research question should identify:

    • the proposed mechanism
    • the limitation it addresses
    • the expected outcome

    Avoid broad questions that could apply to an entire research field.

    17. Return ONE idea for EVERY gap.

    Examples:

    BAD:

        Research Question:
        How effective are current RAG architectures?

        Hypothesis:
        Current RAG architectures have different strengths.

        Methodology:
        Compare existing RAG architectures.

    GOOD:

        Research Question:
        Can an adaptive retrieval orchestration framework dynamically select retrieval strategies based on query complexity to improve factual consistency?

        Hypothesis:
        Selecting retrieval strategies according to query characteristics will improve factual accuracy while reducing unnecessary retrieval operations.

        Methodology:
        Develop a retrieval orchestration module that predicts query complexity and dynamically selects among existing retrieval strategies using current evaluation settings.

    --------------------------------------------------

    BAD:

        Research Question:
        Can RAG be used in healthcare?

    GOOD:

        Research Question:
        Can domain-specific retrieval confidence estimation improve factual reliability of RAG systems in healthcare knowledge bases?

    --------------------------------------------------

    Before producing the final answer, verify:

    ✓ Could this idea have been written without reading the validated gap?

    If YES,
    rewrite it so that it explicitly depends on the validated gap evidence.

    Generic ideas will be considered incorrect.

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