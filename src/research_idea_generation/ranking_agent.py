"""
ranking_agent.py

Phase 7 - Step 3

Ranking Agent

Input:
    List[ScoredIdea]

Output:
    List[RankedIdea]
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict

from typing import List

from src.research_idea_generation.schemas import (
    ScoredIdea,
    RankedIdea,
)


class RankingAgent:

    def __init__(
        self,
        graph_weight: float = 0.20,
        gap_weight: float = 0.20,
        novelty_weight: float = 0.25,
        impact_weight: float = 0.20,
        feasibility_weight: float = 0.10,
        competition_weight: float = 0.05,
    ):

        self.graph_weight = graph_weight
        self.gap_weight = gap_weight
        self.novelty_weight = novelty_weight
        self.impact_weight = impact_weight
        self.feasibility_weight = feasibility_weight
        self.competition_weight = competition_weight

    # =====================================================
    # Score one idea
    # =====================================================

    def _compute_score(
        self,
        idea: ScoredIdea,
    ) -> float:

        competition_component = (
            1.0 - idea.competition_score
        )

        score = (

            self.graph_weight
            * idea.graph_confidence

            +

            self.gap_weight
            * idea.gap_confidence

            +

            self.novelty_weight
            * idea.novelty_score

            +

            self.impact_weight
            * idea.impact_score

            +

            self.feasibility_weight
            * idea.feasibility_score

            +

            self.competition_weight
            * competition_component

        )

        return round(score, 4)

    # =====================================================
    # Rank all ideas
    # =====================================================

    def rank(
        self,
        ideas: List[ScoredIdea]
    ) -> List[RankedIdea]:

        ranked = []

        for idea in ideas:

            final_score = self._compute_score(
                idea
            )

            ranked.append(

                RankedIdea(

                    gap_id=idea.gap_id,

                    graph_confidence=idea.graph_confidence,

                    gap_confidence=idea.gap_confidence,

                    research_question=idea.research_question,

                    hypothesis=idea.hypothesis,

                    proposed_methodology=idea.proposed_methodology,

                    supporting_explanation=idea.supporting_explanation,

                    expected_contribution=idea.expected_contribution,

                    research_direction=idea.research_direction,

                    competition_score=idea.competition_score,

                    impact_score=idea.impact_score,

                    novelty_score=idea.novelty_score,

                    feasibility_score=idea.feasibility_score,

                    competition_reason=idea.competition_reason,

                    impact_reason=idea.impact_reason,

                    novelty_reason=idea.novelty_reason,

                    feasibility_reason=idea.feasibility_reason,

                    final_score=final_score,

                    rank=0

                )

            )

        ranked.sort(

            key=lambda x: x.final_score,

            reverse=True

        )

        for idx, idea in enumerate(

            ranked,

            start=1

        ):

            idea.rank = idx

        output_path = Path(
            "data/processed/ranked_ideas.json"
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
                [asdict(x) for x in ranked],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved {len(ranked)} ranked ideas to {output_path}"
        )

        return ranked


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from src.research_idea_generation.gap_loader import GapLoader

    from src.research_idea_generation.hypothesis_generator import (
        HypothesisGenerator,
    )

    from src.research_idea_generation.novelty_scorer import (
        NoveltyScorer,
    )

    loader = GapLoader(

        candidate_file="data/processed/final_gap_candidates.json",

        gap_file="data/processed/research_gaps.json",

    )

    gaps = loader.load()

    generator = HypothesisGenerator()

    scorer = NoveltyScorer()

    ideas = generator.generate(gaps)

    scored_ideas = scorer.score(
        ideas,
        gaps,
    )

    ranker = RankingAgent()

    ranked = ranker.rank(scored_ideas)

    print("\n========== FINAL RANKING ==========\n")

    for idea in ranked:

        print(

            f"Rank {idea.rank} | "

            f"{idea.gap_id} | "

            f"{idea.final_score:.4f} | "

            f"{idea.research_direction}"

        )