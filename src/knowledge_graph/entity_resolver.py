"""
Entity resolution and normalization for the knowledge graph.

Handles duplicate detection, entity registration, and text normalization
for all semantic entity types.
"""

import re
from typing import Optional

from src.knowledge_graph.graph_utils import (
    generate_author_id,
    generate_venue_id,
    normalize_name,
)

from src.knowledge_graph.schema import (
    NODE_TYPES,
    AUTHOR_PREFIX,
    VENUE_PREFIX,
)


class EntityResolver:
    """
    Manages entity deduplication and normalization across semantic entity types.

    Maintains registries for each entity type (Method, Model, Algorithm, etc.)
    and provides normalization rules to handle variations of the same entity.

    Normalization rules:
        - Lowercase all text
        - Strip leading/trailing whitespace
        - Collapse multiple spaces into single space
        - Replace spaces with underscores for IDs
        - Remove punctuation where appropriate
    """

    def __init__(self) -> None:
        """Initialize entity registries for each semantic entity type."""
        # Initialize a registry for each entity type: {normalized_name -> node_id}
        self._registries: dict[str, dict[str, str]] = {entity_type: {} for entity_type in NODE_TYPES}
        self.authors: dict[str, str] = {}
        self.venues: dict[str, str] = {}

    def normalize(self, text: str) -> str:
        """Normalize text for semantic entity matching."""
        return normalize_name(text)

    def entity_exists(self, entity_type: str, normalized_name: str) -> bool:
        """
        Check if an entity is already registered.

        Args:
            entity_type: Type of entity (e.g., "METHOD", "DATASET").
            normalized_name: Normalized entity name.

        Returns:
            True if entity exists in registry, False otherwise.

        Raises:
            ValueError: If entity_type is not a valid node type.
        """
        if entity_type not in self._registries:
            raise ValueError(
                f"Invalid entity type '{entity_type}'. "
                f"Must be one of: {sorted(NODE_TYPES)}"
            )

        return normalized_name in self._registries[entity_type]

    def register_entity(self, entity_type: str, normalized_name: str, node_id: str) -> None:
        """
        Register an entity in the resolver.

        Associates a normalized entity name with its node ID for deduplication
        in subsequent entity extractions.

        Args:
            entity_type: Type of entity (e.g., "METHOD", "DATASET").
            normalized_name: Normalized entity name (should already be normalized).
            node_id: Unique identifier for the entity node.

        Raises:
            ValueError: If entity_type is not a valid node type.
            KeyError: If entity is already registered with a different node_id.
        """
        if entity_type not in self._registries:
            raise ValueError(
                f"Invalid entity type '{entity_type}'. "
                f"Must be one of: {sorted(NODE_TYPES)}"
            )

        if normalized_name in self._registries[entity_type]:
            existing_id = self._registries[entity_type][normalized_name]
            if existing_id != node_id:
                raise KeyError(
                    f"Entity '{normalized_name}' of type '{entity_type}' "
                    f"already registered with ID '{existing_id}', "
                    f"cannot register with different ID '{node_id}'."
                )
        else:
            self._registries[entity_type][normalized_name] = node_id

    def get_entity(self, entity_type: str, normalized_name: str) -> Optional[str]:
        """
        Retrieve a registered entity's node ID.

        Args:
            entity_type: Type of entity (e.g., "METHOD", "DATASET").
            normalized_name: Normalized entity name.

        Returns:
            The node_id if entity exists, None otherwise.

        Raises:
            ValueError: If entity_type is not a valid node type.
        """
        if entity_type not in self._registries:
            raise ValueError(
                f"Invalid entity type '{entity_type}'. "
                f"Must be one of: {sorted(NODE_TYPES)}"
            )

        return self._registries[entity_type].get(normalized_name)

    def get_all_entities(self, entity_type: str) -> dict[str, str]:
        """
        Retrieve all registered entities of a specific type.

        Args:
            entity_type: Type of entity (e.g., "METHOD", "DATASET").

        Returns:
            Dictionary mapping normalized names to node IDs.

        Raises:
            ValueError: If entity_type is not a valid node type.
        """
        if entity_type not in self._registries:
            raise ValueError(
                f"Invalid entity type '{entity_type}'. "
                f"Must be one of: {sorted(NODE_TYPES)}"
            )

        return self._registries[entity_type].copy()

    def reset(self) -> None:
        for entity_type in self._registries:
            self._registries[entity_type].clear()

        self.authors.clear()
        self.venues.clear()

    def stats(self) -> dict[str, int]:
        """
        Get counts of registered entities by type.

        Returns:
            Dictionary mapping entity types to count of registered entities.
        """
        stats = {
            entity_type: len(registry)
            for entity_type, registry in self._registries.items()
        }

        stats["Author"] = len(self.authors)
        stats["Venue"] = len(self.venues)

        return stats
    
    def get_or_create_author(self, name: str) -> str:
        """
        Return an existing Author node ID or create a new one.
        """
        normalized = normalize_name(name)

        if normalized in self.authors:
            return self.authors[normalized]

        node_id = generate_author_id(name)
        self.authors[normalized] = node_id
        return node_id


    def get_or_create_venue(self, name: str) -> str:
        """
        Return an existing Venue node ID or create a new one.
        """
        normalized = normalize_name(name)

        if normalized in self.venues:
            return self.venues[normalized]

        node_id = generate_venue_id(name)
        self.venues[normalized] = node_id
        return node_id