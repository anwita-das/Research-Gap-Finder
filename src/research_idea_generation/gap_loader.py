"""
gap_loader.py

Phase 7 - Gap Loader

Loads:
    - Phase 5 candidate gaps
    - Phase 6 validated research gaps

Merges them using gap_id.

Output:
    List[MergedGap]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ==========================================================
# Merged Gap Schema
# ==========================================================

@dataclass
class MergedGap:

    # ---------- Phase 5 ----------

    gap_id: str

    source_node: str

    target_node: str

    relation: str

    shared_entity: str

    motif_score: float

    graphsage_score: float

    graph_confidence: float

    status: str

    # ---------- Phase 6 ----------

    gap_type: str

    title: str

    description: str

    supporting_evidence: list

    comparison_summary: str

    temporal_evidence: dict

    gap_confidence: float


# ==========================================================
# Loader
# ==========================================================

class GapLoader:

    def __init__(
        self,
        candidate_file: str,
        gap_file: str
    ):

        self.candidate_file = Path(candidate_file)

        self.gap_file = Path(gap_file)

    # ------------------------------------------------------

    def load(self) -> List[MergedGap]:

        if not self.candidate_file.exists():
            raise FileNotFoundError(self.candidate_file)

        if not self.gap_file.exists():
            raise FileNotFoundError(self.gap_file)

        with open(
            self.candidate_file,
            "r",
            encoding="utf-8"
        ) as f:

            candidates = json.load(f)

        with open(
            self.gap_file,
            "r",
            encoding="utf-8"
        ) as f:

            gaps = json.load(f)

        # ---------------------------------------------
        # Create lookup table for Phase 6
        # ---------------------------------------------

        gap_lookup = {

            gap["gap_id"]: gap

            for gap in gaps

        }

        merged = []

        # ---------------------------------------------
        # Merge Phase 5 + Phase 6
        # ---------------------------------------------

        for candidate in candidates:

            gap_id = candidate["gap_id"]

            if gap_id not in gap_lookup:
                continue

            gap = gap_lookup[gap_id]

            merged.append(

                MergedGap(

                    # -------------------------
                    # Phase 5
                    # -------------------------

                    gap_id=gap_id,

                    source_node=candidate["source_node"],

                    target_node=candidate["target_node"],

                    relation=candidate["relation"],

                    shared_entity=candidate["shared_entity"],

                    motif_score=float(candidate["motif_score"]),

                    graphsage_score=float(candidate["graphsage_score"]),

                    graph_confidence=float(
                        candidate["confidence"]
                    ),

                    status=candidate["status"],

                    # -------------------------
                    # Phase 6
                    # -------------------------

                    gap_type=gap["gap_type"],

                    title=gap["title"],

                    description=gap["description"],

                    supporting_evidence=gap.get(
                        "supporting_evidence",
                        []
                    ),

                    comparison_summary=gap.get(
                        "comparison_summary",
                        ""
                    ),

                    temporal_evidence=gap.get(
                        "temporal_evidence",
                        {}
                    ),

                    gap_confidence=float(
                        gap.get(
                            "confidence",
                            0.0
                        )
                    )
                )

            )

        print(
            f"[GapLoader] Loaded {len(merged)} validated gaps."
        )

        return merged
    
# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    loader = GapLoader(

        candidate_file=
        "data/processed/final_gap_candidates.json",

        gap_file=
        "data/processed/research_gaps.json"

    )

    gaps = loader.load()

    print()

    print(gaps[0])