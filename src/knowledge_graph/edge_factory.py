"""
Edge factory for connecting Paper nodes to semantic entity nodes.

Provides methods to create edges from Paper nodes to semantic entities
with proper relationship attributes and deduplication.
"""

from typing import Optional, Tuple

import networkx as nx

from src.knowledge_graph.schema import (
    ADDRESSES_TASK,
    BELONGS_TO_FIELD,
    EVALUATED_BY,
    HAS_KEYWORD,
    MAKES_CLAIM,
    USES_ALGORITHM,
    USES_DATASET,
    USES_METHOD,
    USES_MODEL,
)

# Type alias for edge tuple returned by MultiDiGraph
EdgeTuple = Tuple[str, str, str]
"""
(source_node_id, target_node_id, relation)
"""


class EdgeFactory:
    """
    Factory for creating edges between Paper nodes and semantic entity nodes.

    Manages the creation of relationships from Paper nodes to semantic entities
    (Methods, Models, Algorithms, Datasets, Tasks, Metrics, Keywords, Fields, Claims)
    with proper deduplication and attribute management.

    The factory assumes that Paper nodes already exist in the graph with IDs
    formatted as: paper:<paper_id>

    Edge attributes:
        key: The relationship type (for MultiDiGraph identification)
        relation: The relationship type (for semantic meaning)
        weight: Always 1.0

    Example:
        >>> factory = EdgeFactory(graph)
        >>> edge = factory.connect_method("paper:arxiv123", "method:resnet")
        >>> # edge is now in the graph
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        """
        Initialize the edge factory.

        Args:
            graph: The networkx.MultiDiGraph to add edges to.
        """
        self.graph = graph

    @staticmethod
    def _create_edge_attrs(relation: str) -> dict[str, str | float]:
        """
        Create standard edge attributes.

        Args:
            relation: The relationship type.

        Returns:
            Dictionary with key, relation, and weight attributes.
        """
        return {
            "relation": relation,
            "weight": 1.0,
        }

    def _add_edge(
        self,
        paper_id: str,
        semantic_node_id: str,
        relation: str,
    ) -> Optional[EdgeTuple]:
        """
        Add an edge from a Paper node to a semantic entity node.

        Prevents duplicate edges with the same (source, target, relation).
        Returns None if edge already exists.

        Args:
            paper_id: ID of the Paper node (e.g., "paper:arxiv123").
            semantic_node_id: ID of the semantic entity node (e.g., "method:resnet").
            relation: The relationship type (e.g., "USES_METHOD").

        Returns:
            Tuple of (source, target, key) if edge was created, None if already exists.
        """
        # Ensure both nodes exist
        if paper_id not in self.graph:
            raise ValueError(f"Paper node '{paper_id}' does not exist.")

        if semantic_node_id not in self.graph:
            raise ValueError(f"Semantic node '{semantic_node_id}' does not exist.")
        
        # Check if edge already exists
        if self.graph.has_edge(paper_id, semantic_node_id, key=relation):
            return None

        # Create edge attributes
        attrs = self._create_edge_attrs(relation)

        # Add edge to graph
        self.graph.add_edge(
            paper_id,
            semantic_node_id,
            key=relation,
            **attrs,
        )

        return (paper_id, semantic_node_id, relation)

    def connect_method(
        self,
        paper_id: str,
        method_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a USES_METHOD edge from Paper to Method node.

        Args:
            paper_id: ID of the Paper node.
            method_node_id: ID of the Method node.

        Returns:
            Tuple of (paper_id, method_node_id, "USES_METHOD") if created, None if duplicate.
        """
        return self._add_edge(paper_id, method_node_id, USES_METHOD)

    def connect_model(
        self,
        paper_id: str,
        model_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a USES_MODEL edge from Paper to Model node.

        Args:
            paper_id: ID of the Paper node.
            model_node_id: ID of the Model node.

        Returns:
            Tuple of (paper_id, model_node_id, "USES_MODEL") if created, None if duplicate.
        """
        return self._add_edge(paper_id, model_node_id, USES_MODEL)

    def connect_algorithm(
        self,
        paper_id: str,
        algorithm_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a USES_ALGORITHM edge from Paper to Algorithm node.

        Args:
            paper_id: ID of the Paper node.
            algorithm_node_id: ID of the Algorithm node.

        Returns:
            Tuple of (paper_id, algorithm_node_id, "USES_ALGORITHM") if created, None if duplicate.
        """
        return self._add_edge(paper_id, algorithm_node_id, USES_ALGORITHM)

    def connect_dataset(
        self,
        paper_id: str,
        dataset_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a USES_DATASET edge from Paper to Dataset node.

        Args:
            paper_id: ID of the Paper node.
            dataset_node_id: ID of the Dataset node.

        Returns:
            Tuple of (paper_id, dataset_node_id, "USES_DATASET") if created, None if duplicate.
        """
        return self._add_edge(paper_id, dataset_node_id, USES_DATASET)

    def connect_task(
        self,
        paper_id: str,
        task_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create an ADDRESSES_TASK edge from Paper to Task node.

        Args:
            paper_id: ID of the Paper node.
            task_node_id: ID of the Task node.

        Returns:
            Tuple of (paper_id, task_node_id, "ADDRESSES_TASK") if created, None if duplicate.
        """
        return self._add_edge(paper_id, task_node_id, ADDRESSES_TASK)

    def connect_metric(
        self,
        paper_id: str,
        metric_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create an EVALUATED_BY edge from Paper to Metric node.

        Args:
            paper_id: ID of the Paper node.
            metric_node_id: ID of the Metric node.

        Returns:
            Tuple of (paper_id, metric_node_id, "EVALUATED_BY") if created, None if duplicate.
        """
        return self._add_edge(paper_id, metric_node_id, EVALUATED_BY)

    def connect_keyword(
        self,
        paper_id: str,
        keyword_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a HAS_KEYWORD edge from Paper to Keyword node.

        Args:
            paper_id: ID of the Paper node.
            keyword_node_id: ID of the Keyword node.

        Returns:
            Tuple of (paper_id, keyword_node_id, "HAS_KEYWORD") if created, None if duplicate.
        """
        return self._add_edge(paper_id, keyword_node_id, HAS_KEYWORD)

    def connect_field(
        self,
        paper_id: str,
        field_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a BELONGS_TO_FIELD edge from Paper to Field node.

        Args:
            paper_id: ID of the Paper node.
            field_node_id: ID of the Field node.

        Returns:
            Tuple of (paper_id, field_node_id, "BELONGS_TO_FIELD") if created, None if duplicate.
        """
        return self._add_edge(paper_id, field_node_id, BELONGS_TO_FIELD)

    def connect_claim(
        self,
        paper_id: str,
        claim_node_id: str,
    ) -> Optional[EdgeTuple]:
        """
        Create a MAKES_CLAIM edge from Paper to Claim node.

        Args:
            paper_id: ID of the Paper node.
            claim_node_id: ID of the Claim node.

        Returns:
            Tuple of (paper_id, claim_node_id, "MAKES_CLAIM") if created, None if duplicate.
        """
        return self._add_edge(paper_id, claim_node_id, MAKES_CLAIM)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def edge_exists(
        self,
        paper_id: str,
        semantic_node_id: str,
        relation: str,
    ) -> bool:
        """
        Check if an edge already exists.

        Args:
            paper_id: ID of the Paper node.
            semantic_node_id: ID of the semantic entity node.
            relation: The relationship type.

        Returns:
            True if edge exists, False otherwise.
        """
        return self.graph.has_edge(paper_id, semantic_node_id, key=relation)

    def get_edge_data(
        self,
        paper_id: str,
        semantic_node_id: str,
        relation: str,
    ) -> Optional[dict]:
        """
        Get edge attributes.

        Args:
            paper_id: ID of the Paper node.
            semantic_node_id: ID of the semantic entity node.
            relation: The relationship type.

        Returns:
            Dictionary of edge attributes, or None if edge doesn't exist.
        """
        try:
            return dict(self.graph.get_edge_data(paper_id, semantic_node_id, key=relation))
        except (KeyError, TypeError):
            return None

    def get_edges_from_paper(self, paper_id: str) -> list[dict]:
        """
        Get all edges from a Paper node.

        Args:
            paper_id: ID of the Paper node.

        Returns:
            List of dictionaries containing edge information (source, target, relation, data).
        """
        edges = []
        for target, key_dict in self.graph[paper_id].items():
            for key, data in key_dict.items():
                edges.append(
                    {
                        "source": paper_id,
                        "target": target,
                        "relation": data.get("relation", key),
                        "data": data,
                    }
                )
        return edges

    def get_edges_by_relation(self, paper_id: str, relation: str) -> list[str]:
        """
        Get all target nodes connected to a Paper by a specific relation.

        Args:
            paper_id: ID of the Paper node.
            relation: The relationship type.

        Returns:
            List of target node IDs.
        """
        targets = []
        for target, key_dict in self.graph[paper_id].items():
            if relation in key_dict:
                targets.append(target)
        return targets

    def edge_count_from_paper(self, paper_id: str) -> int:
        """
        Get the number of edges from a Paper node.

        Args:
            paper_id: ID of the Paper node.

        Returns:
            Number of edges.
        """
        return self.graph.out_degree(paper_id)
