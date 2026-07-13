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

import attrs
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
    attrs = graph.nodes[node_id]
    return str(attrs.get("label", "")).lower() == "paper"

def _is_entity_node(node_id: str, graph: nx.Graph) -> bool:
    attrs = graph.nodes[node_id]
    node_type = str(attrs.get("label", "")).lower()
    return node_type in {
        "method", "dataset", "task", "keyword",
        "model", "metric", "claim", "algorithm"
    }

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

    for middle in graph.nodes():

        connected_papers = []

        # papers pointing to middle
        for node in graph.predecessors(middle):
            if _is_paper_node(node, graph):
                connected_papers.append(node)

        # papers pointed by middle
        for node in graph.successors(middle):
            if _is_paper_node(node, graph):
                connected_papers.append(node)


        connected_papers = list(set(connected_papers))


        # two papers sharing the same middle node
        if len(connected_papers) >= 2:

            pairs = len(connected_papers) * (len(connected_papers)-1) // 2

            two_hop_paths += pairs


            if str(graph.nodes[middle].get("label","")).lower() == "author":
                paper_author_paper += pairs


            if _is_entity_node(middle, graph):
                paper_entity_paper += pairs
                semantic_motifs += pairs

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
    """Generate candidate missing edges from shared entity motifs."""

    if graph is None:
        return []

    candidates = []
    seen = set()

    entity_relations = {
        "USES_MODEL",
        "USES_DATASET",
        "USES_METHOD",
        "USES_ALGORITHM",
        "ADDRESSES_TASK",
        "EVALUATED_BY",
        "HAS_KEYWORD"
    }

    for entity in graph.nodes():

        if not _is_entity_node(entity, graph):
            continue

        papers = []

        # Find papers connected to this entity
        for paper in graph.predecessors(entity):

            if _is_paper_node(paper, graph):
                papers.append(paper)


        for paper in graph.successors(entity):

            if _is_paper_node(paper, graph):
                papers.append(paper)


        # Need at least two papers sharing entity
        if len(papers) < 2:
            continue


        # Generate possible missing paper-paper relationships
        for i in range(len(papers)):

            for j in range(i + 1, len(papers)):

                p1 = papers[i]
                p2 = papers[j]

                if graph.has_edge(p1, p2) or graph.has_edge(p2, p1):
                    continue


                key = (p1, p2, "RELATED_TO")

                if key in seen:
                    continue

                seen.add(key)

                candidates.append(
                    {
                        "gap_id": f"gap_{len(candidates)+1:04d}",
                        "source_node": p1,
                        "target_node": p2,
                        "relation": f"SHARES_{graph.nodes[entity].get('label').upper()}",
                        "shared_entity": entity,
                        "motif_score": min(1.0, len(papers)/10),
                        "graphsage_score": 0.0,
                        "confidence": 0.8,
                        "status": "candidate"
                    }
                )


    candidates = candidates[:max_candidates]


    if output_path:
        save_motif_candidates(candidates, output_path)


    return candidates


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

    print("\n===== EDGE RELATIONS =====")

    relations = set()

    for _, _, data in graph.edges(data=True):
        relations.add(data.get("relation"))

    print(relations)
    motifs = detect_motifs(graph)
    candidates = generate_motif_candidates(graph, output_path="data/processed/motif_candidates.json")

    print("Detected motifs:")
    print(json.dumps(motifs, indent=2))
    print("\nGenerated candidates:")
    print(json.dumps(candidates[:20], indent=2))


if __name__ == "__main__":
    main()
