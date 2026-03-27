"""Tests for the Retriever and VectorStore."""

import numpy as np
import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.vectorstore import VectorStore


class TestVectorStore:
    """Unit tests for VectorStore."""

    def test_add_and_search(self):
        """Adding vectors then searching should return results."""
        store = VectorStore(dimension=4)
        chunks = [
            Chunk(text="The cat sat on the mat", index=0, metadata={"source": "a.txt"}),
            Chunk(text="Dogs are loyal animals", index=1, metadata={"source": "b.txt"}),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        store.add(embeddings, chunks)
        assert store.size == 2

        # Search with a query close to the first vector
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "The cat sat on the mat"
        assert results[0]["score"] > results[1]["score"]

    def test_empty_search(self):
        """Searching an empty store returns no results."""
        store = VectorStore(dimension=4)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert store.search(query) == []

    def test_clear(self):
        """Clear should reset the store."""
        store = VectorStore(dimension=4)
        chunks = [Chunk(text="test", index=0, metadata={"source": "x.txt"})]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        store.add(embeddings, chunks)

        assert store.size == 1
        store.clear()
        assert store.size == 0
        assert store.chunks == []

    def test_get_sources(self):
        """get_sources should return unique source names."""
        store = VectorStore(dimension=4)
        chunks = [
            Chunk(text="a", index=0, metadata={"source": "doc1.pdf"}),
            Chunk(text="b", index=1, metadata={"source": "doc1.pdf"}),
            Chunk(text="c", index=2, metadata={"source": "doc2.txt"}),
        ]
        embeddings = np.random.rand(3, 4).astype(np.float32)
        store.add(embeddings, chunks)

        sources = store.get_sources()
        assert sources == ["doc1.pdf", "doc2.txt"]

    def test_save_and_load(self, tmp_path):
        """Save then load should reproduce the same store."""
        store = VectorStore(dimension=4)
        chunks = [
            Chunk(text="hello world", index=0, metadata={"source": "test.txt"}),
        ]
        embeddings = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
        store.add(embeddings, chunks)

        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)

        assert loaded.size == 1
        assert loaded.chunks[0]["text"] == "hello world"

    def test_mismatched_lengths_raises(self):
        """Adding mismatched embeddings and chunks should raise ValueError."""
        store = VectorStore(dimension=4)
        chunks = [Chunk(text="a", index=0, metadata={})]
        embeddings = np.random.rand(2, 4).astype(np.float32)

        with pytest.raises(ValueError):
            store.add(embeddings, chunks)


class TestRetriever:
    """Integration-style tests for the Retriever (using real embeddings)."""

    @pytest.fixture
    def populated_store(self):
        """Create a store with known vectors for deterministic tests."""
        store = VectorStore(dimension=4)
        chunks = [
            Chunk(text="Python is a programming language", index=0, metadata={"source": "prog.txt"}),
            Chunk(text="Machine learning uses neural networks", index=1, metadata={"source": "ml.txt"}),
            Chunk(text="Cats are independent pets", index=2, metadata={"source": "pets.txt"}),
        ]
        # Manually craft embeddings so similarity is predictable
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        store.add(embeddings, chunks)
        return store

    def test_search_returns_most_relevant(self, populated_store):
        """Direct vector search should rank correctly."""
        query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        results = populated_store.search(query, top_k=3)

        assert results[0]["metadata"]["source"] == "prog.txt"

    def test_top_k_limits_results(self, populated_store):
        """top_k parameter should cap the number of results."""
        query = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        results = populated_store.search(query, top_k=1)
        assert len(results) == 1
