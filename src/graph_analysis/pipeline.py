"""Phase 5 graph analysis pipeline.

This module orchestrates the full Phase 5 workflow:
1. Validate/load the existing knowledge graph.
2. Run motif analysis.
3. Run GraphSAGE preprocessing and candidate scoring.
4. Merge motif and GraphSAGE candidates.
5. Save final merged candidate gaps to JSON.

The pipeline is designed to be modular, configurable, and safe to run even when
inputs are missing or the graph is empty.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from src.graph_analysis.candidate_merger import merge_pipeline, save_candidates
from src.graph_analysis.motif_analysis import (
    detect_motifs,
    generate_motif_candidates,
    load_graph as load_motif_graph,
    save_motif_candidates,
)
from src.graph_analysis.graphsage_preprocess import (
    preprocess_and_train,
)
from src.knowledge_graph.graph_loader import load_existing_graph


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


DEFAULT_GRAPH_PATH = Path("data/processed/knowledge_graph.json")
DEFAULT_MOTIF_OUTPUT = Path("data/processed/motif_candidates.json")
DEFAULT_GRAPHSAGE_OUTPUT = Path("data/processed/graphsage_predictions.json")
DEFAULT_FINAL_OUTPUT = Path("data/processed/final_gap_candidates.json")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_graph_safe(graph_path: Optional[Path] = None) -> nx.MultiDiGraph:
    """Load the existing graph and handle missing/invalid inputs gracefully."""
    try:
        graph = load_existing_graph(graph_path)
    except FileNotFoundError:
        logger.warning("Graph file not found: %s", graph_path)
        return nx.MultiDiGraph()
    except Exception as exc:
        logger.exception("Failed to load graph from %s: %s", graph_path, exc)
        return nx.MultiDiGraph()

    if graph is None:
        return nx.MultiDiGraph()

    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)

    return graph


def validate_graph(graph: nx.Graph) -> bool:
    """Check whether the graph is usable for analysis."""
    if graph is None:
        logger.warning("Graph object is None.")
        return False

    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty.")
        return False

    logger.info(
        "Loaded graph with %d nodes and %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return True


def run_motif_stage(
    graph: nx.Graph,
    motif_output: Path,
    *,
    max_candidates: int = 200,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run motif detection and save motif candidates."""
    motifs = detect_motifs(graph)
    candidates = generate_motif_candidates(
        graph,
        output_path=motif_output,
        max_candidates=max_candidates,
    )

    if not motif_output.exists():
        save_motif_candidates(candidates, motif_output)

    logger.info(
        "Motif stage complete: %d candidates saved to %s",
        len(candidates),
        motif_output,
    )
    return motifs, candidates


def run_graphsage_stage(
    graph: nx.Graph,
    candidates: list[dict[str, Any]],
    graphsage_output: Path,
) -> dict[str, Any]:
    """Run GraphSAGE preprocessing/training/scoring and save predictions."""
    if graph is None or graph.number_of_nodes() == 0:
        logger.warning("Skipping GraphSAGE stage because the graph is empty.")
        save_candidates([], graphsage_output)
        return {
            "mapping": {},
            "features_shape": [0, 0],
            "edge_index_shape": [0, 0],
            "split_info": {},
            "predictions_path": str(graphsage_output),
            "prediction_count": 0,
        }

    try:
        result = preprocess_and_train(
            candidates=candidates,
            output_path=graphsage_output,
            graph_obj=graph,
        )
        logger.info(
            "GraphSAGE stage complete: %d predictions saved to %s",
            result.get("prediction_count", 0),
            graphsage_output,
        )
        return result
    except FileNotFoundError:
        logger.warning("GraphSAGE stage failed because an input file was missing.")
    except Exception as exc:
        logger.exception("GraphSAGE stage failed: %s", exc)

    save_candidates([], graphsage_output)
    return {
        "mapping": {},
        "features_shape": [0, 0],
        "edge_index_shape": [0, 0],
        "split_info": {},
        "predictions_path": str(graphsage_output),
        "prediction_count": 0,
    }


def run_merge_stage(
    motif_output: Path,
    graphsage_output: Path,
    final_output: Path,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold: Optional[float] = 0.0,
    top_k: Optional[int] = 100,
) -> list[dict[str, Any]]:
    """Merge motif and GraphSAGE candidates and save the final output."""
    if not motif_output.exists():
        logger.warning("Motif candidates file missing: %s", motif_output)
        save_candidates([], motif_output)

    if not graphsage_output.exists():
        logger.warning("GraphSAGE predictions file missing: %s", graphsage_output)
        save_candidates([], graphsage_output)

    merged = merge_pipeline(
        motif_path=motif_output,
        graphsage_path=graphsage_output,
        output_path=final_output,
        alpha=alpha,
        beta=beta,
        threshold=threshold,
        top_k=top_k,
    )

    logger.info(
        "Merge stage complete: %d final candidates saved to %s",
        len(merged),
        final_output,
    )
    return merged


def run_pipeline(
    graph_path: Optional[Path] = None,
    *,
    motif_output: Path = DEFAULT_MOTIF_OUTPUT,
    graphsage_output: Path = DEFAULT_GRAPHSAGE_OUTPUT,
    final_output: Path = DEFAULT_FINAL_OUTPUT,
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold: Optional[float] = 0.0,
    top_k: Optional[int] = 100,
    max_motif_candidates: int = 200,
) -> dict[str, Any]:
    """Execute the full Phase 5 workflow end-to-end."""
    graph = load_graph_safe(graph_path or DEFAULT_GRAPH_PATH)
    graph_valid = validate_graph(graph)

    if not graph_valid:
        save_candidates([], motif_output)
        save_candidates([], graphsage_output)
        save_candidates([], final_output)
        return {
            "graph_valid": False,
            "graph_nodes": 0,
            "graph_edges": 0,
            "motif_count": 0,
            "motif_candidates": 0,
            "graphsage_prediction_count": 0,
            "final_candidate_count": 0,
            "motif_output": str(motif_output),
            "graphsage_output": str(graphsage_output),
            "final_output": str(final_output),
        }

    motifs, motif_candidates = run_motif_stage(
        graph,
        motif_output,
        max_candidates=max_motif_candidates,
    )

    graphsage_result = run_graphsage_stage(
        graph,
        motif_candidates,
        graphsage_output,
    )

    final_candidates = run_merge_stage(
        motif_output,
        graphsage_output,
        final_output,
        alpha=alpha,
        beta=beta,
        threshold=threshold,
        top_k=top_k,
    )

    return {
        "graph_valid": True,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "motif_count": int(sum(int(v) for v in motifs.values())) if motifs else 0,
        "motif_candidates": len(motif_candidates),
        "graphsage_prediction_count": int(graphsage_result.get("prediction_count", 0)),
        "final_candidate_count": len(final_candidates),
        "motif_output": str(motif_output),
        "graphsage_output": str(graphsage_output),
        "final_output": str(final_output),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.graph_analysis.pipeline",
        description="Run the Phase 5 graph analysis pipeline end-to-end.",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to the existing Phase 3 graph export.",
    )
    parser.add_argument(
        "--motif-output",
        type=Path,
        default=DEFAULT_MOTIF_OUTPUT,
        help="Path to save motif candidates JSON.",
    )
    parser.add_argument(
        "--graphsage-output",
        type=Path,
        default=DEFAULT_GRAPHSAGE_OUTPUT,
        help="Path to save GraphSAGE predictions JSON.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT,
        help="Path to save final merged candidate JSON.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for motif scores in merged confidence.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight for GraphSAGE scores in merged confidence.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum confidence required to keep a candidate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum number of final candidates to keep.",
    )
    parser.add_argument(
        "--max-motif-candidates",
        type=int,
        default=200,
        help="Maximum number of motif candidates to generate.",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info("Starting Phase 5 pipeline.")
    result = run_pipeline(
        graph_path=args.graph_path,
        motif_output=args.motif_output,
        graphsage_output=args.graphsage_output,
        final_output=args.final_output,
        alpha=args.alpha,
        beta=args.beta,
        threshold=args.threshold,
        top_k=args.top_k,
        max_motif_candidates=args.max_motif_candidates,
    )

    logger.info("Pipeline complete.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()