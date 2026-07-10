"""Lookup graph context for retrieved papers from the existing NetworkX graph."""

from __future__ import annotations

from typing import List, Optional

import networkx as nx

from .types import GraphContext, SearchResult


class GraphConnector:
    """Query the existing NetworkX graph for connected entities related to a paper."""

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None) -> None:
        self.graph = graph

    def enrich_result(self, result: SearchResult) -> SearchResult:
        """Populate the graph context for a single retrieved paper."""
        if self.graph is None:
            return result

        node_id = result.paper_id
        if not self.graph.has_node(node_id):
            return result

        authors = []
        methods = []
        datasets = []
        tasks = []
        models = []
        citation_neighbors = []

        for _, successor, edge_data in self.graph.out_edges(node_id, data=True):

            relation = str(edge_data.get("relation", "")).upper()

            attributes = self.graph.nodes[successor]

            entity_name = (
                attributes.get("name")
                or attributes.get("title")
                or successor
            )

            if relation == "AUTHORED_BY":
                authors.append(entity_name)

            elif relation == "USES_METHOD":
                methods.append(entity_name)

            elif relation == "USES_MODEL":
                models.append(entity_name)

            elif relation == "USES_DATASET":
                datasets.append(entity_name)

            elif relation == "ADDRESSES_TASK":
                tasks.append(entity_name)
        for predecessor, _, edge_data in self.graph.in_edges(node_id, data=True):
            relation = str(edge_data.get("relation", "")).upper()

            if relation == "CITES":
                citation_neighbors.append(predecessor)

        result.graph_context = GraphContext(
            paper_id=result.paper_id,
            authors=authors,
            methods=methods,
            datasets=datasets,
            tasks=tasks,
            citation_neighbors=citation_neighbors,
        )
        return result

    def enrich_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Populate graph context for a list of retrieved papers."""
        return [self.enrich_result(result) for result in results]
