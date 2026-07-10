"""Configurable ranking for semantic search results."""

from __future__ import annotations

from typing import List, Optional

from .config import SemanticSearchConfig
from .types import PaperDocument, SearchResult


class RankingService:
    """Re-rank semantic retrieval results using similarity and metadata."""

    def __init__(self, config: Optional[SemanticSearchConfig] = None) -> None:
        self.config = config or SemanticSearchConfig()

    def rank(self, candidates: List[tuple[PaperDocument, float]]) -> List[SearchResult]:
        """Combine semantic similarity with metadata to produce ranked results."""
        results: List[SearchResult] = []
        for document, semantic_score in candidates:
            metadata = document.metadata or {}
            year = metadata.get("year")
            citation_count = metadata.get("citation_count")
            connectivity = metadata.get("connectivity", 0.0)

            normalized_year = 0.0
            if isinstance(year, (int, float)):
                normalized_year = max(0.0, min(1.0, (year - 2010) / 15.0))

            normalized_citations = 0.0
            if isinstance(citation_count, (int, float)):
                normalized_citations = min(1.0, citation_count / 100.0)

            normalized_connectivity = float(connectivity)
            if normalized_connectivity > 1.0:
                normalized_connectivity = 1.0

            final_score = (
                self.config.similarity_weight * semantic_score
                + self.config.year_weight * normalized_year
                + self.config.citation_weight * normalized_citations
                + self.config.connectivity_weight * normalized_connectivity
            )

            results.append(
                SearchResult(
                    paper_id=document.paper_id,
                    title=document.title,
                    abstract=document.abstract,
                    score=final_score,
                    metadata={**metadata, "semantic_similarity": semantic_score},
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results
