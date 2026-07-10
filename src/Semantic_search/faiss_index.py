"""FAISS vector index wrapper for semantic retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import faiss
import numpy as np


class FaissIndex:
    """Persist and query a FAISS index for dense semantic retrieval."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.index_path = index_path
        self.index: Any = None
        self.ids: List[str] = []

    def build(self, vectors: Sequence[Sequence[float]], paper_ids: Sequence[str]) -> None:
        """Create a FAISS index from vectors and paper identifiers."""
        if len(vectors) != len(paper_ids):
            raise ValueError(
                "The number of vectors must match the number of paper IDs."
            )
        if not vectors:
            self.index = None
            self.ids = []
            return

        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D sequence")
        


        dimension = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(matrix)
        self.ids = [str(paper_id) for paper_id in paper_ids]
        print("Indexed vectors:", self.index.ntotal)

    def add(self, vectors: Sequence[Sequence[float]], paper_ids: Sequence[str]) -> None:
        """Append vectors to the existing index."""
        if len(vectors) != len(paper_ids):
            raise ValueError(
                "The number of vectors must match the number of paper IDs."
            )
        if not vectors:
            return

        if self.index is None:
            self.build(vectors, paper_ids)
            return

        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D sequence")

        if matrix.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.index.d}, got {matrix.shape[1]}."
            )

        self.index.add(matrix)
        self.ids.extend(str(paper_id) for paper_id in paper_ids)

    def search(self, query_vector: Sequence[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-k matching paper identifiers and similarity scores."""
        if self.index is None or not self.ids:
            return []

        vector = np.asarray([query_vector], dtype="float32")

        if vector.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.index.d}, got {vector.shape[1]}."
            )
        scores, indices = self.index.search(vector, min(top_k, len(self.ids)))
        results: List[Tuple[str, float]] = []
        for position, score in zip(indices[0], scores[0]):
            if position < 0:
                continue
            paper_id = self.ids[int(position)]
            results.append((paper_id, float(score)))
        return results

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the FAISS index and metadata to disk."""
        target = path or self.index_path
        if target is None:
            raise ValueError("No FAISS index path was provided.")

        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(target_path))

        metadata_path = target_path.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump({"paper_ids": self.ids}, handle, indent=2)
        self.index_path = target_path
        return target_path

    def load(self, path: Optional[Path] = None) -> None:
        """Load a persisted FAISS index and metadata from disk."""
        target = path or self.index_path
        if target is None:
            raise ValueError("No FAISS index path was provided.")

        target_path = Path(target)

        if not target_path.exists():
            self.index = None
            self.ids = []
            self.index_path = target_path
            return

        self.index = faiss.read_index(str(target_path))
        metadata_path = target_path.with_suffix(".json")
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.ids = [str(item) for item in payload.get("paper_ids", [])]
        else:
            self.ids = []
        self.index_path = target_path
