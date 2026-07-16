"""
schemas.py

Data schemas for Phase 7 - Research Idea Generation.

These schemas are shared across:

- Hypothesis Generator
- Novelty Scorer
- Ranking Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ==========================================================
# Phase 7 Step 1
# Hypothesis Generator Output
# ==========================================================

@dataclass
class HypothesisIdea:

    gap_id: str

    graph_confidence: float

    gap_confidence: float

    research_question: str

    hypothesis: str

    proposed_methodology: str

    supporting_explanation: str

    expected_contribution: str

    research_direction: str

# ==========================================================
# Phase 7 Step 2
# Novelty Scorer Output
# ==========================================================

@dataclass
class ScoredIdea(HypothesisIdea):

    competition_score: float

    impact_score: float

    novelty_score: float

    feasibility_score: float

    competition_reason: str

    impact_reason: str

    novelty_reason: str

    feasibility_reason: str

# ==========================================================
# Phase 7 Step 3
# Final Ranked Research Idea
# ==========================================================

@dataclass
class RankedIdea(ScoredIdea):

    final_score: float

    rank: int = 0