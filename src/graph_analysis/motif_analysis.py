"""Phase 5 motif analysis for the existing knowledge graph.

This module works directly on the exported NetworkX graph when possible and only
reconstructs a graph from the JSON export if the graph object was not provided.
It detects recurring local patterns, counts motif support, generates candidate
missing edges, scores them, and writes JSON output suitable for later modules.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx

from src.knowledge_graph.graph_loader import load_existing_graph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_graph(
    graph_path: Optional[Union[str, Path]] = None,
    *,
    graph_obj: Optional[nx.Graph] = None,
) -> nx.MultiDiGraph:
    """Load the existing graph from disk or reuse an in-memory graph object."""
    if graph_obj is not None:
        if isinstance(graph_obj, nx.MultiDiGraph):
            return graph_obj
        if isinstance(graph_obj, nx.DiGraph):
            return nx.MultiDiGraph(graph_obj)
        raise TypeError("graph_obj must be a NetworkX graph")
    return load_existing_graph(graph_path)


def _is_paper_node(node_id: str, graph: nx.Graph) -> bool:
    attrs = graph.nodes.get(node_id, {})
    return str(attrs.get("type", "")).lower() == "paper" or str(attrs.get("label", "")).lower() == "paper"


def _is_entity_node(node_id: str, graph: nx.Graph) -> bool:
    attrs = graph.nodes.get(node_id, {})
    node_type = str(attrs.get("type", "") or attrs.get("label", "")).lower()
    return node_type in {"method", "dataset", "task", "keyword", "field", "model", "metric", "claim"}


def detect_motifs(graph: nx.Graph) -> dict[str, Any]:
    """Detect recurring motif support on the provided graph.

    The implementation focuses on explainable, low-cost patterns that are easy to
    reason about and safe for both MultiDiGraph and DiGraph.
    """
    if graph is None:
        return {
            "triangles": 0,
            "two_hop_paths": 0,
            "paper_author_paper": 0,
            "paper_entity_paper": 0,
            "semantic_motifs": 0,
        }

    motif_counts: Counter[str] = Counter()

    for node in list(graph.nodes()):
        if not graph.has_node(node):
            continue
        neighbors = list(graph.neighbors(node)) if hasattr(graph, "neighbors") else []
        if len(neighbors) < 2:
            continue

        for first in neighbors:
            if first == node:
                continue
            for second in neighbors:
                if second == node or second == first:
                    continue
                if graph.has_edge(first, second):
                    motif_counts["triangles"] += 1

    two_hop_paths = 0
    paper_author_paper = 0
    paper_entity_paper = 0
    semantic_motifs = 0

    for source in graph.nodes():
        for target in graph.successors(source) if hasattr(graph, "successors") else []:
            if source == target:
                continue
            for mid in graph.successors(target) if hasattr(graph, "successors") else []:
                if mid == source:
                    continue
                two_hop_paths += 1
                if _is_paper_node(source, graph) and _is_paper_node(mid, graph):
                    if _is_entity_node(target, graph):
                        paper_entity_paper += 1
                    elif target.startswith("author:"):
                        paper_author_paper += 1
                if _is_paper_node(source, graph) and _is_entity_node(target, graph):
                    semantic_motifs += 1

    motif_counts["two_hop_paths"] = two_hop_paths
    motif_counts["paper_author_paper"] = paper_author_paper
    motif_counts["paper_entity_paper"] = paper_entity_paper
    motif_counts["semantic_motifs"] = semantic_motifs

    return dict(motif_counts)


def _candidate_key(source: str, target: str, relation: str) -> tuple[str, str, str]:
    return (source, target, relation)


def generate_motif_candidates(
    graph: nx.Graph,
    *,
    output_path: Optional[Union[str, Path]] = None,
    max_candidates: int = 200,
) -> list[dict[str, Any]]:
    """Generate candidate missing edges from local motif patterns."""
    if graph is None:
        return []

    candidates: list[dict[str, Any]] = []
    motif_support: defaultdict[tuple[str, str, str], int] = defaultdict(int)

    for source in graph.nodes():
        successors = list(graph.successors(source)) if hasattr(graph, "successors") else []
        for target in successors:
            if source == target:
                continue
            edge_attrs = graph.get_edge_data(source, target)
            if edge_attrs is None:
                continue
            for _, attrs in edge_attrs.items() if isinstance(edge_attrs, dict) else []:
                relation = attrs.get("relation") or attrs.get("type") or "RELATED"
                if not relation:
                    continue
                if not _is_paper_node(source, graph) and not _is_entity_node(source, graph):
                    continue
                if not _is_paper_node(target, graph) and not _is_entity_node(target, graph):
                    continue
                for other in list(graph.successors(target)) if hasattr(graph, "successors") else []:
                    if other == source or other == target:
                        continue
                    if not graph.has_edge(source, other):
                        candidate_relation = relation
                        # Prefer a semantic relation if the intermediate node is entity-like.
                        if _is_entity_node(target, graph):
                            candidate_relation = "RELATED_TO"
                        key = _candidate_key(source, other, candidate_relation)
                        motif_support[key] += 1
                        candidates.append(
                            {
                                "gap_id": f"gap_{len(candidates) + 1:04d}",
                                "source_node": source,
                                "target_node": other,
                                "relation": candidate_relation,
                                "motif_score": round(min(1.0, motif_support[key] / 8.0), 4),
                                "graphsage_score": round(min(1.0, motif_support[key] / 16.0), 4),
                                "confidence": round(min(1.0, motif_support[key] / 12.0), 4),
                                "status": "candidate",
                            }
                        )

    # Deduplicate and keep a stable, explainable ordering.
    unique_candidates: list[dict[str, Any]] = []
    unique_keys: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate["source_node"], candidate["target_node"], candidate["relation"])
        if key in unique_keys:
            continue
        unique_keys.add(key)
        unique_candidates.append(candidate)

    unique_candidates.sort(key=lambda item: (-float(item["motif_score"]), item["source_node"], item["target_node"]))
    unique_candidates = unique_candidates[:max_candidates]

    if output_path is not None:
        save_motif_candidates(unique_candidates, output_path)

    return unique_candidates


def save_motif_candidates(candidates: list[dict[str, Any]], output_path: Union[str, Path]) -> Path:
    """Persist motif candidates to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    """Run motif analysis and save candidate gaps from the existing graph."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    graph = load_graph()
    print("\n===== SAMPLE NODES =====")
    for i, (node, attrs) in enumerate(graph.nodes(data=True)):
        print(node, "->", attrs)
        if i >= 10:
            break
    motifs = detect_motifs(graph)
    candidates = generate_motif_candidates(graph, output_path="data/processed/motif_candidates.json")

    print("Detected motifs:")
    print(json.dumps(motifs, indent=2))
    print("\nGenerated candidates:")
    print(json.dumps(candidates[:20], indent=2))


if __name__ == "__main__":
    main()
