from typing import Any, Dict, Set
import networkx as nx
from .schema import (
    AUTHOR_PREFIX,
    AuthorProperties,
    NodeLabel,
    PAPER_PREFIX,
    PaperProperties,
    PLACEHOLDER_PROPERTY,
    RelationshipType,
    VenueProperties,
)

class GraphValidationError(ValueError):
    pass

class GraphValidator:
    """Validate nodes, edges, and overall graph structure."""

    def validate_node(self, node_id: str, attrs: Dict[str, Any]) ->None:
        """Validate a single node's attributes."""

        if not node_id:
            raise GraphValidationError("Node identifier cannot be empty.")
        if "label" not in attrs:
            raise GraphValidationError(f"Node '{node_id}' missing required attribute 'label'.")
        
        label = attrs["label"]
        if label not in {label.value for label in NodeLabel}:
            raise GraphValidationError(
                f"Node '{node_id}' has invalid label '{label}'."
            )
        
        if attrs.get("node_id") and attrs["node_id"] != node_id:
            raise GraphValidationError(
            f"Node '{node_id}' has inconsistent node_id attribute'{attrs['node_id']}'."
        )

        if label == NodeLabel.PAPER.value:
            self._validate_paper_node(node_id, attrs)
        elif label == NodeLabel.AUTHOR.value:
            self._validate_author_node(node_id, attrs)
        elif label == NodeLabel.VENUE.value:
            self._validate_venue_node(node_id, attrs)

    def validate_edge(self, u: Any, v: Any, key: Any, attrs: Dict[str,Any]) -> None:
        """Validate a single edge's attributes."""
        if "relation" not in attrs:
            raise GraphValidationError(
                f"Edge {u}->{v} (key={key}) missing required attribute'relation'."
        )
        relation = attrs["relation"]
        if relation not in {rel.value for rel in RelationshipType}:
            raise GraphValidationError(
                f"Edge {u}->{v} (key={key}) has invalid relationship'{relation}'."
            )
        if "weight" not in attrs:
            raise GraphValidationError(
                f"Edge {u}->{v} (key={key}) missing required attribute 'weight'."
            )

        weight = attrs["weight"]
        if not isinstance(weight, (int, float)):
            raise GraphValidationError(
                f"Edge {u}->{v} (key={key}) has non-numeric weight '{weight}'."
            )
        if "key" not in attrs:
            raise GraphValidationError(
                f"Edge {u}->{v} (key={key}) missing required attribute 'key'."
        )

    def validate_graph(self, graph: nx.MultiDiGraph) -> None:
        """Validate the overall graph structure and attributes."""
        self._validate_unique_node_ids(graph)
        self._validate_nodes(graph)
        self._validate_edges(graph)

    def _validate_paper_node(self, node_id: str, attrs: Dict[str, Any]) -> None:
        if attrs.get(PLACEHOLDER_PROPERTY):
            return
        required_fields = {
        PaperProperties.NODE_ID.value,
        PaperProperties.LABEL.value,
        PaperProperties.TITLE.value,
        PaperProperties.ABSTRACT.value,
        PaperProperties.YEAR.value,
        PaperProperties.DOI.value,
        PaperProperties.ARXIV_ID.value,
        PaperProperties.OPENALEX_ID.value,
        PaperProperties.URL.value,
        PaperProperties.PDF_URL.value,
        PaperProperties.CITATION_COUNT.value,
        PaperProperties.REFERENCE_COUNT.value,
        PaperProperties.PUBLICATION_DATE.value,
        PaperProperties.UPDATED_AT.value,
        PaperProperties.SOURCE.value,
        PaperProperties.SUMMARY.value,
        }
        missing = required_fields - set(attrs)
        if missing:
            raise GraphValidationError(
            f"Paper node '{node_id}' is missing required attributes: {sorted(missing)}."
            )
        
    def _validate_author_node(self, node_id: str, attrs: Dict[str,Any]) -> None:
        required_fields = {
        AuthorProperties.NODE_ID.value,
        AuthorProperties.LABEL.value,
        AuthorProperties.NAME.value,
        AuthorProperties.NORMALIZED_NAME.value,
        }
        missing = required_fields - set(attrs)
        if missing:
            raise GraphValidationError(
                f"Author node '{node_id}' is missing required attributes: {sorted(missing)}."
            )
    def _validate_venue_node(self, node_id: str, attrs: Dict[str, Any])-> None:
        required_fields = {
        VenueProperties.NODE_ID.value,
        VenueProperties.LABEL.value,
        VenueProperties.NAME.value,
        VenueProperties.TYPE.value,
        }
        missing = required_fields - set(attrs)
        if missing:
            raise GraphValidationError(
            f"Venue node '{node_id}' is missing required attributes: {sorted(missing)}."
            )
    def _validate_unique_node_ids(self, graph: nx.MultiDiGraph) ->None:
        seen: Set[str] = set()
        for node_id, attrs in graph.nodes(data=True):
            attr_node_id = attrs.get("node_id")
            if attr_node_id is None:
                continue
            if attr_node_id in seen:
                raise GraphValidationError(
                    f"Duplicate node_id attribute found: '{attr_node_id}'."
                )
            seen.add(attr_node_id)
    def _validate_nodes(self, graph: nx.MultiDiGraph) -> None:
        for node_id, attrs in graph.nodes(data=True):
            self.validate_node(node_id, attrs)
    def _validate_edges(self, graph: nx.MultiDiGraph) -> None:
        for u, v, key, attrs in graph.edges(keys=True, data=True):
            if not graph.has_node(u):
                raise GraphValidationError(f"Edge source node '{u}' does not exist.")
            if not graph.has_node(v):
                raise GraphValidationError(f"Edge target node '{v}' does not exist.")
        self.validate_edge(u, v, key, attrs)