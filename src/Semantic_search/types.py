"""Shared data structures for the semantic search subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperDocument:
    """Representation of a paper prepared for embedding and retrieval."""

    paper_id: str
    title: str
    abstract: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    entity_names: List[str] = field(default_factory=list)


@dataclass
class SearchQuery:
    """User query submitted to the semantic search service."""

    text: str
    top_k: Optional[int] = None


@dataclass
class GraphContext:
    paper_id: str
    authors: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    citation_neighbors: List[str] = field(default_factory=list)

@dataclass
class SearchResult:
    """A semantically retrieved paper enriched with graph context."""

    paper_id: str
    title: str
    abstract: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_context: Optional[GraphContext] = None
