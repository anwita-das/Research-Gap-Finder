"""
Knowledge graph builder for constructing semantic entity graphs.

Manages graph construction, entity deduplication, and node management
using networkx.MultiDiGraph as the underlying data structure.
"""

from typing import Any, Optional, Dict

import networkx as nx

from src.knowledge_graph.edge_factory import EdgeFactory
from src.knowledge_graph.entity_resolver import EntityResolver
from src.knowledge_graph.node_factory import NodeFactory
from src.knowledge_graph.graph_utils import generate_paper_id
from src.knowledge_graph.schema import (
    ALGORITHM,
    CLAIM,
    DATASET,
    FIELD,
    KEYWORD,
    METHOD,
    METRIC,
    MODEL,
    TASK,
    PLACEHOLDER_PROPERTY,
)


class GraphBuilder:
    """
    Builder for constructing knowledge graphs with semantic entities.

    Manages a networkx.MultiDiGraph containing semantic entity nodes.
    Handles entity deduplication, node creation, and graph management.

    The builder ensures that no duplicate semantic entities are added to the graph
    by maintaining an EntityResolver that tracks all registered entities.

    Attributes:
        graph: The underlying networkx.MultiDiGraph storing semantic entities.
        _resolver: EntityResolver for managing entity deduplication.

    Example:
        >>> builder = GraphBuilder()
        >>> method_id = builder.add_method("Attention Mechanism")
        >>> dataset_id = builder.add_dataset("ImageNet")
        >>> print(builder.node_count())
        2
    """

    def __init__(self) -> None:
        """Initialize the graph builder with empty graph and resolver."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

        self._resolver: EntityResolver = EntityResolver()

        self.node_factory = NodeFactory()

        self.edge_factory = EdgeFactory(self.graph)

        self.placeholder_papers: set[str] = set()

    def add_paper(self, paper_dict: Dict[str, Any]) -> None:
        """Add a paper and its related author, venue, and citation data."""
        paper_id = str(paper_dict.get("paper_id", "")).strip()
        if not paper_id:
            raise ValueError("paper_dict must include a paper_id")
        
        paper_node_id = self._paper_node_id(paper_id)
        self._ensure_paper_node(paper_node_id, paper_dict)

        for author_name in paper_dict.get("authors", []):
            self.add_authorship(paper_id, author_name)

        venue_name = paper_dict.get("venue")
        venue_type = paper_dict.get("venue_type", "")


        if venue_name:
            self.add_publication(
                paper_id,
                venue_name,
                venue_type,
            )
        for cited_id in paper_dict.get("references", []):
            if cited_id:
                self.add_citation(paper_id, str(cited_id))

    def add_author(self, name: str) -> str:
        """Add an author node if missing and return its node identifier."""
        author_node_id = self._resolver.get_or_create_author(name)
        if not self.graph.has_node(author_node_id):
            attributes = self.node_factory.create_author_node(name)
            self.graph.add_node(author_node_id, **attributes)
        return author_node_id
    
    def add_venue(self, name: str, venue_type: str = "") -> str:
        """Add a venue node if missing and return its node
        identifier."""
        venue_node_id = self._resolver.get_or_create_venue(name)
        if not self.graph.has_node(venue_node_id):
            attributes = self.node_factory.create_venue_node(name,venue_type)
            self.graph.add_node(venue_node_id, **attributes)
        return venue_node_id
    
    def add_authorship(self, paper_id: str, author_name: str) -> None:
        """Connect an author to a paper with an AUTHORED edge."""
        author_node_id = self.add_author(author_name)
        paper_node_id = self._ensure_paper_placeholder(paper_id)
        edge_attrs = self.edge_factory.create_authored_edge()
        edge_key = edge_attrs.pop("key")

        if not self.graph.has_edge(
            author_node_id,
            paper_node_id,
        ):
            self.graph.add_edge(
                author_node_id,
                paper_node_id,
                key=edge_key,
                **edge_attrs,
            )

    def add_citation(self, citing_paper_id: str, cited_paper_id: str)-> None:
        """Connect a citing paper to a cited paper with a CITES edge."""

        citing_node_id = self._ensure_paper_placeholder(citing_paper_id)
        cited_node_id = self._ensure_paper_placeholder(cited_paper_id)
        edge_attrs = self.edge_factory.create_cites_edge()
        edge_key = edge_attrs.pop("key")

        if not self.graph.has_edge(
            citing_node_id,
            cited_node_id,
        ):
            self.graph.add_edge(
                citing_node_id,
                cited_node_id,
                key=edge_key,
                **edge_attrs,
            )

    def add_publication(
        self,
        paper_id: str,
        venue_name: str,
        venue_type: str = "",
    ) -> None:
        """Connect a paper to a venue with a PUBLISHED_IN edge."""

        paper_node_id = self._ensure_paper_placeholder(paper_id)
        venue_node_id = self.add_venue(venue_name, venue_type,)
        edge_attrs = self.edge_factory.create_published_in_edge()
        edge_key = edge_attrs.pop("key")

        if not self.graph.has_edge(
            paper_node_id,
            venue_node_id,
        ):
            self.graph.add_edge(
                paper_node_id,
                venue_node_id,
                key=edge_key,
                **edge_attrs,
            )

    def build_graph(self) -> nx.MultiDiGraph:
        """Return the constructed graph."""
        return self.graph
    
    def _paper_node_id(self, paper_id: str) -> str:
        paper_id = str(paper_id).strip()
        if paper_id.startswith("paper:"):
            return paper_id
        return generate_paper_id(paper_id)
    
    def _ensure_paper_node(self, paper_node_id: str, paper_dict:
        Dict[str, Any]) -> None:
        if self.graph.has_node(paper_node_id):
            if self.graph.nodes[paper_node_id].get(PLACEHOLDER_PROPERTY):
                attributes = self.node_factory.create_paper_node(paper_dict)
                self.graph.nodes[paper_node_id].update(attributes)
                
                self.graph.nodes[paper_node_id].pop(PLACEHOLDER_PROPERTY, None)
                self.placeholder_papers.discard(paper_node_id)
            return
        attributes = self.node_factory.create_paper_node(paper_dict)
        self.graph.add_node(paper_node_id, **attributes)

    def _ensure_paper_placeholder(self, paper_id: str) -> str:
        paper_node_id = self._paper_node_id(str(paper_id))
        if not self.graph.has_node(paper_node_id):
            placeholder_attrs = self.node_factory.create_placeholder_paper_node(paper_id)
            self.graph.add_node(paper_node_id, **placeholder_attrs)
            self.placeholder_papers.add(paper_node_id)
        return paper_node_id

    def _add_entity_node(
        self,
        entity_type: str,
        name: str,
    ) -> str:
        """
        Add or retrieve an entity node.

        Internal method that handles the standard entity node creation workflow:
        1. Normalize the entity name
        2. Check if entity already exists using EntityResolver
        3. Reuse existing node if present
        4. Create new node using NodeFactory
        5. Add to graph with full node attributes
        6. Register in resolver
        7. Return node ID

        Args:
            entity_type: Type of entity (e.g., "METHOD", "DATASET").
            name: Original entity name.

        Returns:
            The node_id of the entity node.
        """

        # Validate input
        if not name or not name.strip():
            raise ValueError("Entity name cannot be empty.")

        # Step 1: Normalize
        normalized = self._resolver.normalize(name)

        # Step 2 & 3: Check duplicates and reuse if exists
        if self._resolver.entity_exists(entity_type, normalized):
            existing_node = self._resolver.get_entity(entity_type, normalized)
            assert existing_node is not None
            return existing_node

        # Step 4: Create new node
        node = self.node_factory.create_entity(entity_type, name, normalized)
        node_id = node["node_id"]

        # Step 5: Insert into graph with full node data
        self.graph.add_node(node_id, **node)

        # Step 6: Register in resolver
        self._resolver.register_entity(entity_type, normalized, node_id)

        return node_id

    def add_method(self, name: str) -> str:
        """
        Add a Method node to the graph.

        If a method with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the research method (e.g., "Cross-Validation").

        Returns:
            The node_id of the method node (e.g., "method:cross_validation").
        """
        return self._add_entity_node(METHOD, name)

    def add_model(self, name: str) -> str:
        """
        Add a Model node to the graph.

        If a model with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the machine learning model (e.g., "ResNet-50").

        Returns:
            The node_id of the model node (e.g., "model:resnet_50").
        """
        return self._add_entity_node(MODEL, name)

    def add_algorithm(self, name: str) -> str:
        """
        Add an Algorithm node to the graph.

        If an algorithm with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the algorithm (e.g., "Gradient Descent").

        Returns:
            The node_id of the algorithm node (e.g., "algorithm:gradient_descent").
        """
        return self._add_entity_node(ALGORITHM, name)

    def add_dataset(self, name: str) -> str:
        """
        Add a Dataset node to the graph.

        If a dataset with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the dataset (e.g., "ImageNet").

        Returns:
            The node_id of the dataset node (e.g., "dataset:imagenet").
        """
        return self._add_entity_node(DATASET, name)

    def add_task(self, name: str) -> str:
        """
        Add a Task node to the graph.

        If a task with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the research task (e.g., "Image Classification").

        Returns:
            The node_id of the task node (e.g., "task:image_classification").
        """
        return self._add_entity_node(TASK, name)

    def add_metric(self, name: str) -> str:
        """
        Add a Metric node to the graph.

        If a metric with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the evaluation metric (e.g., "F1-Score").

        Returns:
            The node_id of the metric node (e.g., "metric:f1_score").
        """
        return self._add_entity_node(METRIC, name)

    def add_keyword(self, name: str) -> str:
        """
        Add a Keyword node to the graph.

        If a keyword with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: The keyword (e.g., "Transfer Learning").

        Returns:
            The node_id of the keyword node (e.g., "keyword:transfer_learning").
        """
        return self._add_entity_node(KEYWORD, name)

    def add_field(self, name: str) -> str:
        """
        Add a Field node to the graph.

        If a field with the same normalized name already exists, returns
        the existing node ID without creating a duplicate.

        Args:
            name: Name of the research field (e.g., "Computer Vision").

        Returns:
            The node_id of the field node (e.g., "field:computer_vision").
        """
        return self._add_entity_node(FIELD, name)

    def add_claim(self, text: str) -> str:
        """
        Add a Claim node to the graph.

        Claim nodes are identified by SHA1 hash of normalized text, ensuring
        that identical claims (after normalization) are deduplicated.

        Args:
            text: The claim text (e.g., "Attention mechanisms improve performance").

        Returns:
            The node_id of the claim node (e.g., "claim:a1b2c3d4...").
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Claim text cannot be empty.")

        # Normalize claim text
        normalized_text = self._resolver.normalize(text)

        # Check if claim already exists
        if self._resolver.entity_exists(CLAIM, normalized_text):
            existing_node = self._resolver.get_entity(CLAIM, normalized_text)
            assert existing_node is not None
            return existing_node

        # Create claim node using NodeFactory
        node = self.node_factory.create_claim(text, normalized_text)
        node_id = node["node_id"]

        # Add to graph with full claim data
        self.graph.add_node(node_id, **node)

        # Register in resolver
        self._resolver.register_entity(CLAIM, normalized_text, node_id)

        return node_id

    # ========================================================================
    # High-level Integration Methods
    # ========================================================================

    def add_semantic_entities_to_paper(
        self,
        paper_id: str,
        extracted_entities: dict[str, Any],
    ) -> dict[str, int]:
        """
        Add extracted semantic entities to a Paper node and create relationships.

        Processes extracted entities from an NLP pipeline and:
        1. Creates or reuses semantic entity nodes for each entity
        2. Connects the Paper node to each semantic entity using appropriate relationships
        3. Gracefully handles None values, empty strings, duplicates, and empty lists

        The extracted_entities dict should have the form:
        {
            "methods": ["method1", "method2", ...],
            "models": ["model1", "model2", ...],
            "algorithms": [...],
            "datasets": [...],
            "tasks": [...],
            "metrics": [...],
            "keywords": [...],
            "fields": [...],
            "claims": [...]
        }

        Args:
            paper_id: ID of the Paper node (e.g., "paper:arxiv123").
                     Assumes the Paper node already exists in the graph.
            extracted_entities: Dictionary of extracted semantic entities by type.

        Returns:
            Summary dict with counts of entities added:
            {
                "methods_added": 2,
                "models_added": 1,
                "algorithms_added": 0,
                "datasets_added": 3,
                "tasks_added": 1,
                "metrics_added": 2,
                "keywords_added": 5,
                "fields_added": 1,
                "claims_added": 2,
                "total": 17
            }

        Example:
            >>> builder = GraphBuilder()
            >>> extracted = {
            ...     "methods": ["Cross-Validation", "Grid Search"],
            ...     "datasets": ["ImageNet"],
            ...     "claims": ["CNNs improve accuracy"]
            ... }
            >>> summary = builder.add_semantic_entities_to_paper("paper:arxiv123", extracted)
            >>> print(summary["total"])
            4
        """

        normalized_paper_id = str(paper_id).strip()
        if not normalized_paper_id:
            raise ValueError("paper_id cannot be empty")
        if not normalized_paper_id.startswith("paper:"):
            normalized_paper_id = self._paper_node_id(normalized_paper_id)

        paper_node_id = self._ensure_paper_placeholder(normalized_paper_id)

        # Initialize summary counters
        summary: dict[str, int] = {
            "methods_added": 0,
            "models_added": 0,
            "algorithms_added": 0,
            "datasets_added": 0,
            "tasks_added": 0,
            "metrics_added": 0,
            "keywords_added": 0,
            "fields_added": 0,
            "claims_added": 0,
            "total": 0,
        }

        # Helper to process entity list for a specific entity type
        def process_entity_type(
            entity_list: Any,
            entity_type: str,
            add_method,
            connect_method,
            summary_key: str,
        ) -> None:
            """Process a list of entities for a specific entity type."""
            # Validate entity_list
            if not entity_list or not isinstance(entity_list, list):
                return

            # Track seen values to avoid duplicates
            seen = set()

            for entity_value in entity_list:
                # Skip None and empty strings
                if not entity_value or not isinstance(entity_value, str):
                    continue

                # Strip whitespace
                entity_value = entity_value.strip()

                # Skip empty after stripping
                if not entity_value:
                    continue

                # Skip duplicates within this list
                normalized_value = self._resolver.normalize(entity_value)

                if normalized_value in seen:
                    continue

                seen.add(normalized_value)

                node_id = add_method(entity_value)

                edge = connect_method(paper_node_id, node_id)

                if edge is not None:
                    summary[summary_key] += 1
                    summary["total"] += 1

        # Process each entity type
        process_entity_type(
            extracted_entities.get("methods"),
            METHOD,
            self.add_method,
            self.edge_factory.connect_method,
            "methods_added",
        )

        process_entity_type(
            extracted_entities.get("models"),
            MODEL,
            self.add_model,
            self.edge_factory.connect_model,
            "models_added",
        )

        process_entity_type(
            extracted_entities.get("algorithms"),
            ALGORITHM,
            self.add_algorithm,
            self.edge_factory.connect_algorithm,
            "algorithms_added",
        )

        process_entity_type(
            extracted_entities.get("datasets"),
            DATASET,
            self.add_dataset,
            self.edge_factory.connect_dataset,
            "datasets_added",
        )

        process_entity_type(
            extracted_entities.get("tasks"),
            TASK,
            self.add_task,
            self.edge_factory.connect_task,
            "tasks_added",
        )

        process_entity_type(
            extracted_entities.get("metrics"),
            METRIC,
            self.add_metric,
            self.edge_factory.connect_metric,
            "metrics_added",
        )

        process_entity_type(
            extracted_entities.get("keywords"),
            KEYWORD,
            self.add_keyword,
            self.edge_factory.connect_keyword,
            "keywords_added",
        )

        process_entity_type(
            extracted_entities.get("fields"),
            FIELD,
            self.add_field,
            self.edge_factory.connect_field,
            "fields_added",
        )

        # Process claims separately (different signature)
        process_entity_type(
            extracted_entities.get("claims"),
            CLAIM,
            self.add_claim,
            self.edge_factory.connect_claim,
            "claims_added",
        )

        return summary

    # ========================================================================
    # Query and Utility Methods
    # ========================================================================

    def get_node(self, node_id: str) -> Optional[dict]:
        """
        Retrieve node attributes by node_id.

        Args:
            node_id: The ID of the node.

        Returns:
            Dictionary of node attributes, or None if node doesn't exist.
        """
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def node_exists(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: The ID of the node.

        Returns:
            True if node exists, False otherwise.
        """
        return node_id in self.graph

    def get_nodes_by_type(self, node_type: str) -> list[str]:
        """
        Get all node IDs of a specific type.

        Args:
            node_type: The node label type (e.g., "METHOD", "DATASET").

        Returns:
            List of node IDs matching the type.
        """
        return [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("label") == node_type
        ]

    def node_count(self) -> int:
        """
        Get the total number of nodes in the graph.

        Returns:
            Number of nodes.
        """
        return self.graph.number_of_nodes()

    def type_counts(self) -> dict[str, int]:
        """
        Get count of nodes by type.

        Returns:
            Dictionary mapping node type label to count of nodes.
        """
        counts: dict[str, int] = {}
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("label")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def resolver_stats(self) -> dict[str, int]:
        """
        Get entity registration statistics from the resolver.

        Returns:
            Dictionary mapping entity type to count of registered entities.
        """
        return self._resolver.stats()