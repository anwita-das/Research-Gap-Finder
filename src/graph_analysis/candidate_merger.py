"""Phase 5 candidate merging.

Combines motif-based and GraphSAGE-based candidate gaps into a single ranked
candidate list for downstream research gap analysis.

This module does not perform motif detection or GraphSAGE inference.
It only merges, scores, filters, and saves candidate gaps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_candidates(
    path: Union[str, Path]
) -> list[dict[str, Any]]:
    """
    Load candidate gaps from a JSON file.

    Expected format:

    [
        {
            "source_node": "...",
            "target_node": "...",
            "relation": "RELATED_TO",
            "motif_score": 0.8,
            "graphsage_score": 0.5
        }
    ]
    """

    file_path = Path(path)

    if not file_path.exists():
        logger.warning(
            "Candidate file does not exist: %s",
            file_path
        )
        return []

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected list of candidates in {file_path}"
        )

    return data


def candidate_key(
    candidate: dict[str, Any]
) -> tuple[str, str, str]:
    """
    Unique identifier for a candidate edge.
    """

    return (
        str(candidate.get("source_node", "")),
        str(candidate.get("target_node", "")),
        str(candidate.get("relation", "RELATED_TO")),
    )


def merge_candidate_lists(
    motif_candidates: list[dict[str, Any]],
    graphsage_candidates: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Merge motif and GraphSAGE candidates.

    Candidates appearing in both sources are combined.
    Candidates appearing in only one source are kept.

    Final confidence:

        confidence =
            alpha * motif_score +
            beta * graphsage_score

    """

    merged: dict[
        tuple[str, str, str],
        dict[str, Any]
    ] = {}


    # Add motif candidates
    for candidate in motif_candidates:

        key = candidate_key(candidate)

        merged[key] = {
            **candidate,
            "motif_score": float(
                candidate.get(
                    "motif_score",
                    0.0
                )
            ),
            "graphsage_score": 0.0,
        }


    # Add / update GraphSAGE candidates
    for candidate in graphsage_candidates:

        key = candidate_key(candidate)

        if key not in merged:

            merged[key] = {
                **candidate,
                "motif_score": 0.0,
                "graphsage_score": float(
                    candidate.get(
                        "graphsage_score",
                        0.0
                    )
                ),
            }

        else:

            merged[key]["graphsage_score"] = float(
                candidate.get(
                    "graphsage_score",
                    0.0
                )
            )


    final_candidates = []


    for candidate in merged.values():

        motif_score = float(
            candidate.get(
                "motif_score",
                0.0
            )
        )

        graphsage_score = float(
            candidate.get(
                "graphsage_score",
                0.0
            )
        )


        confidence = (
            alpha * motif_score
            +
            beta * graphsage_score
        )


        candidate["confidence"] = round(
            min(confidence, 1.0),
            4
        )

        candidate["status"] = "candidate"

        final_candidates.append(candidate)


    return final_candidates



def filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Filter candidates by confidence and optionally keep top-k results.
    """

    filtered = candidates


    if threshold is not None:

        filtered = [
            candidate
            for candidate in filtered
            if candidate.get(
                "confidence",
                0.0
            ) >= threshold
        ]


    filtered.sort(
        key=lambda x: x.get(
            "confidence",
            0.0
        ),
        reverse=True
    )


    if top_k is not None:

        filtered = filtered[:top_k]


    return filtered



def save_candidates(
    candidates: list[dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """
    Save final candidate gaps to JSON.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            candidates,
            file,
            indent=2,
            ensure_ascii=False
        )


    return path



def merge_pipeline(
    motif_path: Union[str, Path],
    graphsage_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Complete candidate merging workflow.
    """

    motif_candidates = load_candidates(
        motif_path
    )

    graphsage_candidates = load_candidates(
        graphsage_path
    )


    merged = merge_candidate_lists(
        motif_candidates,
        graphsage_candidates,
        alpha=alpha,
        beta=beta,
    )


    merged = filter_candidates(
        merged,
        threshold=threshold,
        top_k=top_k,
    )


    save_candidates(
        merged,
        output_path
    )


    return merged



def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )


    output = merge_pipeline(
        motif_path="data/processed/motif_candidates.json",
        graphsage_path="data/processed/graphsage_predictions.json",
        output_path="data/processed/final_gap_candidates.json",
        alpha=0.5,
        beta=0.5,
        threshold=0.0,
        top_k=100,
    )


    print(
        json.dumps(
            output[:10],
            indent=2
        )
    )



if __name__ == "__main__":
    main()