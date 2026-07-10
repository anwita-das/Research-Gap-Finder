"""Configuration settings for the semantic search subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SemanticSearchConfig:
    """Configuration for embedding, indexing, and ranking behavior."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k: int = 10
    index_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    graph_path: Optional[Path] = None
    data_dir: Optional[Path] = None
    similarity_weight: float = 0.7
    year_weight: float = 0.1
    citation_weight: float = 0.1
    connectivity_weight: float = 0.1

    def resolve_paths(self, project_root: Optional[Path] = None) -> "SemanticSearchConfig":
        """Resolve default paths for persisted artifacts and graph input."""
        root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]

        data_dir = root / "data" / "processed" / "semantic_search"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = data_dir
        self.index_path = self.index_path or data_dir / "faiss_index.faiss"
        self.metadata_path = self.metadata_path or data_dir / "paper_ids.json"
        self.graph_path = self.graph_path or root / "data" / "processed" / "knowledge_graph.json" 
        return self
    
