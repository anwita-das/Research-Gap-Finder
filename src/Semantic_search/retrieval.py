"""Semantic retrieval orchestration over FAISS and document metadata."""

from __future__ import annotations

from typing import List, Optional

from networkx import hits

from .config import SemanticSearchConfig
from .embedding_service import EmbeddingService
from .faiss_index import FaissIndex
from .types import PaperDocument, SearchQuery


class SemanticRetriever:
    """Generate embeddings, query FAISS, and return candidate papers."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        faiss_index: Optional[FaissIndex] = None,
        config: Optional[SemanticSearchConfig] = None,
    ) -> None:
        self.config = config or SemanticSearchConfig()
        self.embedding_service = embedding_service or EmbeddingService(self.config)
        self.faiss_index = faiss_index or FaissIndex(self.config.index_path)

    def build_index(self, documents: List[PaperDocument]) -> None:
        """Create a FAISS index from a set of paper documents."""
        if not documents:
            self.faiss_index.build([], [])
            return

        texts = [document.text for document in documents]
        vectors = self.embedding_service.embed_documents(texts)
        
        paper_ids = [document.paper_id for document in documents]
        print("Number of vectors:", len(vectors))
        print("Number of paper IDs:", len(paper_ids))
        self.faiss_index.build(vectors, paper_ids)
        self.faiss_index.save(self.config.index_path)

    def retrieve(self, query: SearchQuery, documents: Optional[List[PaperDocument]] = None) -> List[PaperDocument]:
        """Retrieve semantically similar papers for a query."""
        if self.faiss_index.index is None:
            if documents is None:
                return []
            self.build_index(documents)

        if self.faiss_index.index is None or not self.faiss_index.ids:
                return []

        query_vector = self.embedding_service.embed_text(query.text)
        hits = self.faiss_index.search(query_vector, top_k=query.top_k or self.config.top_k)
        print("Hits from FAISS:")
        print(hits)
        paper_lookup = {document.paper_id: document for document in documents or []}
        retrieved = []
        for paper_id, score in hits:
            document = paper_lookup.get(paper_id)
            if document is None:
                continue
            updated_document = PaperDocument(
                paper_id=document.paper_id,
                title=document.title,
                abstract=document.abstract,
                text=document.text,
                metadata={**document.metadata, "semantic_similarity": score},
                entity_names=document.entity_names,
            )
            print(updated_document.title)
            retrieved.append(updated_document)
            print("Retrieved:", updated_document.title)
        return retrieved
