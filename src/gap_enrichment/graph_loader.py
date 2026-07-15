"""
graph_loader.py

Loads the Phase 3 knowledge graph and provides helper
functions for querying papers and semantic entities.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


class GraphLoader:
    """
    Loads the exported knowledge graph and exposes
    helper methods for Phase 6.
    """

    def __init__(self, graph_file: str):

        self.graph_file = Path(graph_file)

        with open(self.graph_file, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        self.graph = json_graph.node_link_graph(
            graph_data,
            directed=True,
            multigraph=True
        )

        print(
            f"[GraphLoader] Loaded graph with "
            f"{self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges."
        )

    # ---------------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        """
        Check whether a node exists.
        """

        return self.graph.has_node(node_id)

    # ---------------------------------------------------------

    def get_node(self, node_id: str):
        """
        Return node attributes.
        """

        if not self.graph.has_node(node_id):
            return None

        return self.graph.nodes[node_id]

    # ---------------------------------------------------------

    def get_neighbors(self, node_id: str):
        """
        Return neighboring nodes.

        Works for both paper nodes and entity nodes.
        """

        if not self.graph.has_node(node_id):
            return []

        neighbors = set()

        neighbors.update(self.graph.successors(node_id))
        neighbors.update(self.graph.predecessors(node_id))

        return list(neighbors)

    # ---------------------------------------------------------

    def get_connected_papers(self, entity_node: str):
        """
        Return every paper connected to an entity.

        Example
        -------
        task:question_answering
            ->
        [
            paper:arxiv_...
            paper:arxiv_...
        ]
        """

        if not self.graph.has_node(entity_node):
            return []

        papers = set()

        for neighbor in self.get_neighbors(entity_node):

            if neighbor.startswith("paper:"):
                papers.add(neighbor)

        return sorted(papers)

    # ---------------------------------------------------------

    def get_paper_entities(self, paper_node: str):
        """
        Return every semantic entity connected
        to a paper.
        """

        if not self.graph.has_node(paper_node):
            return []

        entities = []

        for neighbor in self.get_neighbors(paper_node):

            if not neighbor.startswith("paper:"):
                entities.append(neighbor)

        return sorted(entities)

    # ---------------------------------------------------------

    def get_relation(self, source: str, target: str):
        """
        Return edge data between two nodes.

        Useful for debugging.
        """

        if self.graph.has_edge(source, target):
            return self.graph.get_edge_data(source, target)

        return None

    # ---------------------------------------------------------

    def get_graph(self):
        """
        Return the underlying NetworkX graph.
        """

        return self.graph


# ---------------------------------------------------------

if __name__ == "__main__":

    loader = GraphLoader(
        "data/processed/knowledge_graph.json"
    )

    entity = "task:question_answering"

    print("\nConnected papers:")
    print(loader.get_connected_papers(entity))