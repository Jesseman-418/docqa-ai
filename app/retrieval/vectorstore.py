"""FAISS vector store management with persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


class VectorStore:
    """Manages a FAISS index paired with chunk metadata.

    Supports adding new embeddings, searching by vector, and
    persisting / loading the index from disk.
    """

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.embedding.dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[dict] = []  # parallel list of serialised chunk metadata

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and their associated chunks to the store.

        Args:
            embeddings: Array of shape (n, dimension).
            chunks: Corresponding Chunk objects (same length as embeddings).
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length.")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        for chunk in chunks:
            self.chunks.append(
                {
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                }
            )

        logger.info("Added %d vectors. Total index size: %d", len(chunks), self.size)

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
        """Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: 1-D array of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, score, metadata.
        """
        if self.size == 0:
            return []

        top_k = min(top_k or settings.retrieval.top_k, self.size)
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_data = self.chunks[idx]
            results.append(
                {
                    "text": chunk_data["text"],
                    "score": float(score),
                    "metadata": chunk_data["metadata"],
                }
            )
        return results

    def save(self, directory: Path | str | None = None) -> Path:
        """Persist the FAISS index and metadata to disk.

        Args:
            directory: Target directory. Defaults to ``settings.index_dir``.

        Returns:
            The directory path used.
        """
        directory = Path(directory or settings.index_dir)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / _INDEX_FILE))
        (directory / _META_FILE).write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2))

        logger.info("Saved index (%d vectors) to %s", self.size, directory)
        return directory

    @classmethod
    def load(cls, directory: Path | str | None = None) -> "VectorStore":
        """Load a previously saved index from disk.

        Args:
            directory: Directory containing index files. Defaults to ``settings.index_dir``.

        Returns:
            A populated VectorStore instance.
        """
        directory = Path(directory or settings.index_dir)
        index_path = directory / _INDEX_FILE
        meta_path = directory / _META_FILE

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No saved index found in {directory}")

        store = cls()
        store.index = faiss.read_index(str(index_path))
        store.chunks = json.loads(meta_path.read_text())
        store.dimension = store.index.d

        logger.info("Loaded index (%d vectors) from %s", store.size, directory)
        return store

    def clear(self) -> None:
        """Reset the index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        logger.info("Cleared vector store.")

    def get_sources(self) -> list[str]:
        """Return a deduplicated list of source filenames in the store."""
        seen: set[str] = set()
        sources: list[str] = []
        for c in self.chunks:
            src = c["metadata"].get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources
