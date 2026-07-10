"""Read-only loading of the existing exported knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
from networkx.readwrite import json_graph


class GraphLoader:
    """Load the existing knowledge graph export without modifying graph structure."""

    def __init__(self, graph_path: Optional[Path] = None) -> None:
        self.graph_path = graph_path
        self.graph: Optional[nx.MultiDiGraph] = None

    def load_graph(self, graph_path: Optional[Path] = None) -> nx.MultiDiGraph:
        """Load a NetworkX graph from the existing exported graph file."""
        path = Path(graph_path or self.graph_path or Path(__file__).resolve().parents[2] / "data" / "processed" / "knowledge_graph.json")
        if self.graph is not None and self.graph_path == path:
            return self.graph

        self.graph_path = path
        if not path.exists():
            self.graph = nx.MultiDiGraph()
            return self.graph

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict) and ("nodes" in payload or "links" in payload):
            self.graph = json_graph.node_link_graph(payload, directed=True, multigraph=True)
        elif isinstance(payload, dict) and "graph" in payload:
            self.graph = json_graph.adjacency_graph(payload["graph"])
        else:
            self.graph = nx.MultiDiGraph()
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes")
        print(f"Loaded graph with {self.graph.number_of_edges()} edges")

        return self.graph

    def load_paper_nodes(self) -> list[Dict[str, Any]]:
        graph = self.graph or self.load_graph()
        paper_nodes = []

        for node_id, attributes in graph.nodes(data=True):
            label = attributes.get("label") or attributes.get("node_label")

            if str(label).lower() != "paper":
                continue

            # Skip placeholder papers
            if attributes.get("placeholder", False):
                continue

            paper_nodes.append({"node_id": node_id, **attributes})
        print("Paper nodes found:", len(paper_nodes))
        return paper_nodes
