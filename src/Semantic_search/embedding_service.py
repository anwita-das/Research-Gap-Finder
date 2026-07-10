"""Embedding generation for semantic retrieval using sentence-transformers."""

from __future__ import annotations

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .config import SemanticSearchConfig


class EmbeddingService:
    """Generate dense embeddings for paper documents and user queries."""

    def __init__(self, config: Optional[SemanticSearchConfig] = None) -> None:
        self.config = config or SemanticSearchConfig()
        self.model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """Load the sentence-transformers model specified by the configuration."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.config.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.config.model_name}'"
                ) from exc

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a dense vector."""
        if not isinstance(text, str) or not text.strip():
            return [0.0] * self.config.embedding_dim

        self.load_model()
        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.astype(float).tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a collection of text documents."""
        if not documents:
            return []

        self.load_model()
        embeddings = self.model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(float).tolist()
