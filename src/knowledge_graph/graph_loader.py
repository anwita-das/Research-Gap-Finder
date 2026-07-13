"""Load and validate the existing exported knowledge graph without rebuilding it unnecessarily."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

REQUIRED_NODE_ATTRIBUTES = ("node_id", "type", "name", "properties")
REQUIRED_EDGE_ATTRIBUTES = ("edge_id", "source", "target", "relation", "weight")
SCHEMA_VERSION = "phase5_v1"

def _resolve_graph_path(graph_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a graph path from an explicit input or the default processed export."""
    if graph_path is None:
        return Path(__file__).resolve().parents[2] / "data" / "processed" / "knowledge_graph.json"
    return Path(graph_path)


def load_graph_payload(graph_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load the graph JSON payload safely and return a dictionary when possible."""
    path = _resolve_graph_path(graph_path)
    if not path.exists():
        logger.warning("Graph file does not exist: %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse graph JSON %s: %s", path, exc)
        return {}
    except OSError as exc:
        logger.warning("Could not read graph JSON %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Graph payload at %s is not a JSON object", path)
        return {}

    payload.setdefault("schema_version", SCHEMA_VERSION)
    return payload


def _normalize_node_payload(node_payload: dict[str, Any], *, default_node_id: Optional[str] = None) -> dict[str, Any]:
    """Normalize node payloads from the existing JSON export to the Phase 5 schema."""
    normalized = dict(node_payload)
    node_id = str(normalized.get("node_id") or normalized.get("id") or default_node_id or "").strip()
    if not node_id:
        raise ValueError("Encountered a node without a usable node_id")

    normalized.setdefault("node_id", node_id)
    normalized.setdefault("type", normalized.get("type") or normalized.get("label") or "unknown")
    normalized.setdefault("name", normalized.get("name") or normalized.get("title") or normalized.get("text") or normalized.get("normalized_name") or node_id)

    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        properties = {
            key: value
            for key, value in normalized.items()
            if key not in {"node_id", "id", "type", "name", "properties"}
        }
    normalized["properties"] = properties
    return normalized


def _normalize_edge_payload(edge_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize edge payloads from the existing JSON export to the Phase 5 schema."""
    normalized = dict(edge_payload)
    relation = normalized.get("relation") or normalized.get("type") or normalized.get("label") or "RELATED"
    source = normalized.get("source") or normalized.get("from") or normalized.get("u")
    target = normalized.get("target") or normalized.get("to") or normalized.get("v")
    edge_key = normalized.get("key") or normalized.get("edge_id") or normalized.get("id")
    edge_id = str(edge_key or f"{source}->{target}:{relation}")

    weight = normalized.get("weight", 1.0)

    normalized["edge_id"] = edge_id
    normalized["key"] = edge_id
    normalized["source"] = source
    normalized["target"] = target
    normalized["relation"] = relation
    normalized["weight"] = weight
    return normalized


def reconstruct_graph_from_payload(payload: dict[str, Any]) -> nx.MultiDiGraph:
    """Rebuild a NetworkX MultiDiGraph from a node-link payload if one is present."""
    graph = nx.MultiDiGraph()
    nodes_payload = payload.get("nodes")
    edges_payload = payload.get("edges")

    if not isinstance(nodes_payload, list) or not isinstance(edges_payload, list):
        logger.warning("Graph payload does not contain a list of nodes and edges")
        return graph

    for node_payload in nodes_payload:
        if not isinstance(node_payload, dict):
            continue
        try:
            normalized_node = _normalize_node_payload(node_payload)
        except ValueError as exc:
            logger.warning("Skipping invalid node payload: %s", exc)
            continue

        node_id = normalized_node["node_id"]
        graph.add_node(node_id, **normalized_node)

    for edge_payload in edges_payload:
        if not isinstance(edge_payload, dict):
            continue

        normalized_edge = _normalize_edge_payload(edge_payload)
        source = normalized_edge.get("source")
        target = normalized_edge.get("target")
        if not source or not target:
            logger.warning("Skipping edge with missing endpoints: %s", edge_payload)
            continue

        edge_key = normalized_edge["edge_id"]
        edge_attrs = dict(normalized_edge)
        edge_attrs.pop("key", None)
        graph.add_edge(source, target, key=edge_key, **edge_attrs)

    return graph


def load_existing_graph(
    graph_path: Optional[Union[str, Path]] = None,
    *,
    graph_obj: Optional[nx.MultiDiGraph] = None,
) -> nx.MultiDiGraph:
    """Load an existing graph from a path if available; otherwise reconstruct from JSON."""
    if graph_obj is not None:
        if not isinstance(graph_obj, nx.MultiDiGraph):
            raise TypeError("graph_obj must be a networkx.MultiDiGraph")
        return graph_obj

    payload = load_graph_payload(graph_path)
    if not payload:
        return nx.MultiDiGraph()

    if isinstance(payload.get("nodes"), list) and isinstance(payload.get("edges"), list):
        return reconstruct_graph_from_payload(payload)

    if isinstance(payload.get("graph"), dict):
        try:
            return nx.node_link_graph(payload, directed=True, multigraph=True)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Could not reconstruct graph from node-link payload: %s", exc)
            return nx.MultiDiGraph()


def get_graph_statistics(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Return reusable graph summary statistics for downstream modules."""
    if graph is None:
        return {"node_count": 0, "edge_count": 0, "directed": True, "multigraph": True, "average_degree": 0.0, "isolated_nodes": 0}

    directed = getattr(graph, "is_directed", lambda: True)()
    multigraph = getattr(graph, "is_multigraph", lambda: True)()
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if node_count:
        average_out_degree = edge_count / node_count
        average_in_degree = edge_count / node_count
    else:
        average_out_degree = 0.0
        average_in_degree = 0.0

    isolated_nodes = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "directed": directed,
        "multigraph": multigraph,
        "average_out_degree": round(average_out_degree, 4),
        "average_in_degree": round(average_in_degree, 4),
        "isolated_nodes": isolated_nodes,
    }


def get_missing_attribute_report(graph: nx.MultiDiGraph) -> dict[str, list[dict[str, Any]]]:
    """Report nodes and edges whose required Phase 5 attributes are missing."""
    report: dict[str, list[dict[str, Any]]] = {"nodes": [], "edges": []}

    for node_id, attrs in graph.nodes(data=True):
        missing = []
        for attr_name in REQUIRED_NODE_ATTRIBUTES:
            value = attrs.get(attr_name)
            if attr_name == "properties":
                if not isinstance(value, dict):
                    missing.append(attr_name)
            elif value is None or value == "":
                missing.append(attr_name)
        if missing:
            report["nodes"].append({"node_id": node_id, "missing_attributes": missing})

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_id = attrs.get("edge_id") or key or f"{source}->{target}"
        missing = []
        for attr_name in REQUIRED_EDGE_ATTRIBUTES:
            value = attrs.get(attr_name)
            if attr_name == "weight":
                if not isinstance(value, (int, float)):
                    missing.append(attr_name)
            elif value in (None, ""):
                missing.append(attr_name)
        if missing:
            report["edges"].append(
                {
                    "edge_id": edge_id,
                    "source": source,
                    "target": target,
                    "missing_attributes": missing,
                }
            )

    return report


def get_invalid_edge_report(graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Return invalid edge records whose endpoints or attributes are inconsistent."""
    report: list[dict[str, Any]] = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_id = attrs.get("edge_id") or key or f"{source}->{target}"
        issues = []
        if source in (None, ""):
            issues.append("missing_source_node")
        if target in (None, ""):
            issues.append("missing_target_node")
        if attrs.get("relation") in (None, ""):
            issues.append("missing_relation")
        if not isinstance(attrs.get("weight"), (int, float)):
            issues.append("invalid_weight")
        if issues:
            report.append({"edge_id": edge_id, "source": source, "target": target, "issues": issues})
    return report


def validate_graph(
    graph: nx.MultiDiGraph,
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Validate the graph and return structured reports suitable for downstream consumers."""
    statistics = get_graph_statistics(graph)
    missing_report = get_missing_attribute_report(graph)
    invalid_edges = get_invalid_edge_report(graph)

    result = {
        "statistics": statistics,
        "missing_attributes": missing_report,
        "invalid_edges": invalid_edges,
    }

    if raise_on_error and (missing_report["nodes"] or missing_report["edges"] or invalid_edges):
        raise ValueError(
            "Graph validation failed: "
            f"{len(missing_report['nodes'])} node issues, "
            f"{len(missing_report['edges'])} edge issues, "
            f"{len(invalid_edges)} invalid edge issues"
        )

    return result

def get_phase5_readiness_report(graph: nx.MultiDiGraph) -> dict[str, Any]:
    validation = validate_graph(graph)
    duplicate_semantic_edges = {}
    seen = set()

    for source, target, key, attrs in graph.edges(keys=True, data=True):
        relation = attrs.get("relation")
        semantic_key = (source, target, relation)
        duplicate_semantic_edges.setdefault(semantic_key, 0)
        duplicate_semantic_edges[semantic_key] += 1
        seen.add(key)

    duplicates = [
        {"source": s, "target": t, "relation": r, "count": c}
        for (s, t, r), c in duplicate_semantic_edges.items()
        if c > 1
    ]

    return {
        "validation": validation,
        "duplicate_semantic_edges": duplicates,
        "ready_for_motif_analysis": not validation["missing_attributes"]["nodes"] and not validation["missing_attributes"]["edges"] and not validation["invalid_edges"],
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    graph = load_existing_graph()

    print("Statistics:")
    print(json.dumps(get_graph_statistics(graph), indent=2))

    print("\nValidation:")
    print(json.dumps(validate_graph(graph), indent=2))

    print("\nPhase 5 Readiness:")
    print(json.dumps(get_phase5_readiness_report(graph), indent=2))