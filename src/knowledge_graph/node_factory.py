"""
Node factory for creating standardized semantic entity nodes.

Provides methods to create node dictionaries with consistent structure
and ID generation for all semantic entity types.
"""

import hashlib
from typing import Any, TypedDict, Dict

from src.knowledge_graph.graph_utils import (
    generate_author_id,
    generate_paper_id,
    generate_venue_id,
    normalize_name,
)

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
    NodeLabel,
    PaperProperties,
    AuthorProperties,
    VenueProperties,
    PLACEHOLDER_PROPERTY,
)


class BaseNode(TypedDict):
    """Base structure for all nodes."""

    node_id: str
    label: str


class EntityNode(TypedDict):
    """Structure for semantic entity nodes."""

    node_id: str
    label: str
    name: str
    normalized_name: str


class ClaimNode(TypedDict):
    """Structure for claim nodes."""

    node_id: str
    label: str
    text: str
    normalized_text: str


class NodeFactory:
    """
    Factory for creating standardized knowledge graph nodes.

    Generates node dictionaries with consistent structure and ID format.
    Supports creation of Method, Model, Algorithm, Dataset, Task, Metric,
    Keyword, Field, and Claim nodes.

    Node ID format:
        - entity_type:normalized_name (for most entities)
        - claim:sha1_hash (for claims)

    Example:
        >>> factory = NodeFactory()
        >>> node = factory.create_method("ResNet", "resnet")
        >>> node['node_id']
        'method:resnet'
    """

    @staticmethod
    def create_method(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Method node.

        Args:
            name: Original name of the method.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"method:{normalized_name}",
            "label": METHOD,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_model(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Model node.

        Args:
            name: Original name of the model.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"model:{normalized_name}",
            "label": MODEL,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_algorithm(name: str, normalized_name: str) -> EntityNode:
        """
        Create an Algorithm node.

        Args:
            name: Original name of the algorithm.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"algorithm:{normalized_name}",
            "label": ALGORITHM,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_dataset(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Dataset node.

        Args:
            name: Original name of the dataset.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"dataset:{normalized_name}",
            "label": DATASET,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_task(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Task node.

        Args:
            name: Original name of the task.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"task:{normalized_name}",
            "label": TASK,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_metric(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Metric node.

        Args:
            name: Original name of the metric.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"metric:{normalized_name}",
            "label": METRIC,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_keyword(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Keyword node.

        Args:
            name: Original keyword.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"keyword:{normalized_name}",
            "label": KEYWORD,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_field(name: str, normalized_name: str) -> EntityNode:
        """
        Create a Field node.

        Args:
            name: Original name of the field.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.
        """
        return {
            "node_id": f"field:{normalized_name}",
            "label": FIELD,
            "name": name,
            "normalized_name": normalized_name,
        }

    @staticmethod
    def create_claim(text: str, normalized_text: str) -> ClaimNode:
        """
        Create a Claim node.

        Claim node IDs are generated using SHA1 hash of the normalized text
        to ensure uniqueness and consistency across runs.

        Args:
            text: Original claim text.
            normalized_text: Normalized claim text (for hashing).

        Returns:
            Node dictionary with structure: node_id, label, text, normalized_text.
        """
        # Generate SHA1 hash of normalized text
        text_hash = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()

        return {
            "node_id": f"claim:{text_hash}",
            "label": CLAIM,
            "text": text,
            "normalized_text": normalized_text,
        }

    @staticmethod
    def create_entity(
        entity_type: str, name: str, normalized_name: str
    ) -> EntityNode:
        """
        Create an entity node based on type.

        Generic factory method that dispatches to specific entity creators.

        Args:
            entity_type: Type of entity (e.g., "METHOD", "DATASET").
            name: Original name of the entity.
            normalized_name: Normalized name (lowercase, underscores).

        Returns:
            Node dictionary with structure: node_id, label, name.

        Raises:
            ValueError: If entity_type is not supported.
        """
        entity_creators = {
            METHOD: NodeFactory.create_method,
            MODEL: NodeFactory.create_model,
            ALGORITHM: NodeFactory.create_algorithm,
            DATASET: NodeFactory.create_dataset,
            TASK: NodeFactory.create_task,
            METRIC: NodeFactory.create_metric,
            KEYWORD: NodeFactory.create_keyword,
            FIELD: NodeFactory.create_field,
        }

        if entity_type not in entity_creators:
            raise ValueError(
                f"Unsupported entity type '{entity_type}'. "
                f"Must be one of: {sorted(entity_creators.keys())}"
            )

        return entity_creators[entity_type](name, normalized_name)
    
    @staticmethod
    def create_paper_node(paper_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Paper node."""

        paper_id = paper_dict.get("paper_id")

        if not paper_id:
            raise ValueError("paper_dict must include a paper_id")

        metadata = paper_dict.get("metadata", {})

        return {
            PaperProperties.NODE_ID.value: generate_paper_id(str(paper_id)),
            PaperProperties.LABEL.value: NodeLabel.PAPER.value,
            PaperProperties.TITLE.value: paper_dict.get("title", ""),
            PaperProperties.ABSTRACT.value: paper_dict.get("abstract", ""),
            PaperProperties.YEAR.value: paper_dict.get("year"),
            PaperProperties.DOI.value: paper_dict.get("doi", ""),
            PaperProperties.ARXIV_ID.value: paper_dict.get("arxiv_id", ""),
            PaperProperties.OPENALEX_ID.value: paper_dict.get("openalex_id", ""),
            PaperProperties.URL.value: paper_dict.get("url", ""),
            PaperProperties.PDF_URL.value: paper_dict.get("pdf_url", ""),
            PaperProperties.CITATION_COUNT.value: metadata.get("citation_count", 0),
            PaperProperties.REFERENCE_COUNT.value: metadata.get("reference_count", 0),
            PaperProperties.PUBLICATION_DATE.value: metadata.get("publication_date", ""),
            PaperProperties.UPDATED_AT.value: metadata.get("updated_at", ""),
            PaperProperties.SOURCE.value: paper_dict.get("source", []),
            PaperProperties.SUMMARY.value: paper_dict.get("summary", ""),
        }

    @staticmethod
    def create_placeholder_paper_node(paper_id: str) -> Dict[str, Any]:
        """Create a placeholder Paper node."""

        return {
            PaperProperties.NODE_ID.value: generate_paper_id(str(paper_id)),
            PaperProperties.LABEL.value: NodeLabel.PAPER.value,
            PaperProperties.TITLE.value: "",
            PLACEHOLDER_PROPERTY: True,
        }

    @staticmethod
    def create_author_node(name: str) -> Dict[str, Any]:
        """Create an Author node."""

        normalized_name = normalize_name(name)

        return {
            AuthorProperties.NODE_ID.value: generate_author_id(name),
            AuthorProperties.LABEL.value: NodeLabel.AUTHOR.value,
            AuthorProperties.NAME.value: name,
            AuthorProperties.NORMALIZED_NAME.value: normalized_name,
        }

    @staticmethod
    def create_venue_node(
        name: str,
        venue_type: str = "",
    ) -> Dict[str, Any]:
        """Create a Venue node."""

        return {
            VenueProperties.NODE_ID.value: generate_venue_id(name),
            VenueProperties.LABEL.value: NodeLabel.VENUE.value,
            VenueProperties.NAME.value: name,
            VenueProperties.TYPE.value: venue_type,
        }