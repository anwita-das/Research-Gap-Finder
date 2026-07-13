from pathlib import Path

import networkx as nx

from src.knowledge_graph.graph_loader import (
    get_graph_statistics,
    get_invalid_edge_report,
    get_missing_attribute_report,
    load_existing_graph,
)


def test_load_existing_graph_json_and_return_reports():
    graph_path = Path("data/processed/knowledge_graph.json")

    graph = load_existing_graph(graph_path)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    stats = get_graph_statistics(graph)
    assert stats["node_count"] == graph.number_of_nodes()
    assert stats["edge_count"] == graph.number_of_edges()

    missing = get_missing_attribute_report(graph)
    assert set(missing.keys()) == {"nodes", "edges"}
    assert isinstance(missing["nodes"], list)
    assert isinstance(missing["edges"], list)

    invalid_edges = get_invalid_edge_report(graph)
    assert isinstance(invalid_edges, list)
