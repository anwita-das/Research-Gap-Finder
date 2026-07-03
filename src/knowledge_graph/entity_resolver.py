"""
Entity resolution and normalization for the knowledge graph.

Handles duplicate detection, entity registration, and text normalization
for all semantic entity types.
"""

import re
from typing import Optional

from src.knowledge_graph.schema import NODE_TYPES


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

    def normalize(self, text: str) -> str:
        """
        Normalize text for entity matching and ID generation.

        Applies a series of transformations to standardize entity names:
        1. Strip leading/trailing whitespace
        2. Lowercase all characters
        3. Collapse multiple spaces to single space
        4. Replace spaces with underscores (for use in IDs)
        5. Remove non-alphanumeric characters except underscores

        Args:
            text: Raw entity name or text to normalize.

        Returns:
            Normalized text suitable for entity matching and ID generation.

        Example:
            >>> resolver = EntityResolver()
            >>> resolver.normalize("ImageNet")
            'imagenet'
            >>> resolver.normalize(" image net ")
            'image_net'
            >>> resolver.normalize("IMAGENET")
            'imagenet'
        """
        # Strip and lowercase
        normalized = text.strip().lower()

        # Collapse multiple spaces into single space
        normalized = re.sub(r"[^\w\s]", " ", normalized)

        # Replace spaces with underscores
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Remove punctuation and special characters, keep only alphanumeric and underscores
        normalized = normalized.replace(" ", "_")

        return normalized

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
        """Clear all entity registries."""
        for entity_type in self._registries:
            self._registries[entity_type].clear()

    def stats(self) -> dict[str, int]:
        """
        Get counts of registered entities by type.

        Returns:
            Dictionary mapping entity types to count of registered entities.
        """
        return {
            entity_type: len(registry)
            for entity_type, registry in self._registries.items()
        }
