"""
novelty_scorer.py

Phase 7 - Step 2

Novelty Scorer Agent

Input:
    HypothesisIdea

Output:
    ScoredIdea
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict

from src.gap_finding.llm_helper import LLMHelper
from src.research_idea_generation.gap_loader import MergedGap

from src.research_idea_generation.schemas import (
    HypothesisIdea,
    ScoredIdea,
)

from src.research_idea_generation.prompts import (
    build_batch_novelty_prompt,
)


class NoveltyScorer:

    def __init__(self):

        self.llm = LLMHelper()

    # ======================================================
    # MAIN FUNCTION
    # ======================================================

    def score(
        self,
        ideas: list[HypothesisIdea],
        gaps: list[MergedGap],
    ) -> list[ScoredIdea]:

        cached = self.load_cache()

        if cached is not None:

            print(
                f"Loaded {len(cached)} scored ideas from cache."
            )

            return cached

        prompt = build_batch_novelty_prompt(
            ideas,
            gaps
        )

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

            print(
                "Novelty scoring failed:",
                e
            )

            return []

        gap_lookup = {
            gap.gap_id: gap
            for gap in gaps
        }

        idea_lookup = {
            idea.gap_id: idea
            for idea in ideas
        }

        scored = []

        for item in result:

            idea = idea_lookup[item["gap_id"]]

            scored.append(

                ScoredIdea(

                    gap_id=idea.gap_id,

                    graph_confidence=idea.graph_confidence,

                    gap_confidence=idea.gap_confidence,

                    research_question=idea.research_question,

                    hypothesis=idea.hypothesis,

                    proposed_methodology=idea.proposed_methodology,

                    supporting_explanation=idea.supporting_explanation,

                    expected_contribution=idea.expected_contribution,

                    research_direction=idea.research_direction,

                    competition_score=float(
                        item.get("competition_score", 0.5)
                    ),

                    impact_score=float(
                        item.get("impact_score", 0.5)
                    ),

                    novelty_score=float(
                        item.get("novelty_score", 0.5)
                    ),

                    feasibility_score=float(
                        item.get("feasibility_score", 0.5)
                    ),

                    competition_reason=item.get(
                        "competition_reason",
                        "",
                    ),

                    impact_reason=item.get(
                        "impact_reason",
                        "",
                    ),

                    novelty_reason=item.get(
                        "novelty_reason",
                        "",
                    ),

                    feasibility_reason=item.get(
                        "feasibility_reason",
                        "",
                    ),
                )

            )

        output_path = Path(
            "data/processed/scored_ideas.json"
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
                [asdict(x) for x in scored],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved {len(scored)} scored ideas to {output_path}"
        )

        return scored


    def load_cache(self):

        path = Path(
            "data/processed/scored_ideas.json"
        )

        if not path.exists():
            return None

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

        except json.JSONDecodeError:

            return None

        return [
            ScoredIdea(**item)
            for item in data
        ]

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from src.research_idea_generation.gap_loader import GapLoader
    from src.research_idea_generation.hypothesis_generator import (
        HypothesisGenerator,
    )

    loader = GapLoader(

        candidate_file="data/processed/final_gap_candidates.json",

        gap_file="data/processed/research_gaps.json"

    )

    gaps = loader.load()

    generator = HypothesisGenerator()

    ideas = generator.generate(gaps)

    scorer = NoveltyScorer()

    scored = scorer.score(
        ideas,
        gaps
    )

    print()

    print(scored[0])