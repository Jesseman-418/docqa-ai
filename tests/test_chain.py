"""Tests for the QA chain."""

import numpy as np
import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.embeddings import EmbeddingEngine
from app.retrieval.vectorstore import VectorStore
from app.retrieval.retriever import Retriever
from app.qa.chain import QAChain, Answer
from app.qa.prompts import format_context, build_extractive_prompt


class TestPrompts:
    """Tests for prompt formatting."""

    def test_format_context(self):
        """format_context should produce numbered passage blocks."""
        results = [
            {"text": "First passage", "score": 0.95, "metadata": {"source": "a.pdf"}},
            {"text": "Second passage", "score": 0.80, "metadata": {"source": "b.txt"}},
        ]
        output = format_context(results)

        assert "Passage 1" in output
        assert "Passage 2" in output
        assert "a.pdf" in output
        assert "0.95" in output
        assert "First passage" in output

    def test_build_extractive_prompt(self):
        """build_extractive_prompt should include question and context."""
        results = [
            {"text": "Some content", "score": 0.9, "metadata": {"source": "doc.pdf"}},
        ]
        prompt = build_extractive_prompt("What is AI?", results)

        assert "What is AI?" in prompt
        assert "Some content" in prompt
        assert "doc.pdf" in prompt


class TestQAChainExtractive:
    """Tests for the extractive QA chain."""

    @pytest.fixture
    def chain(self):
        """Build a QAChain with a small in-memory store."""
        store = VectorStore(dimension=4)
        chunks = [
            Chunk(text="Artificial intelligence is the simulation of human intelligence by machines.",
                  index=0, metadata={"source": "ai.txt"}),
            Chunk(text="Python was created by Guido van Rossum in 1991.",
                  index=1, metadata={"source": "python.txt"}),
            Chunk(text="Neural networks are a subset of machine learning algorithms.",
                  index=2, metadata={"source": "ml.txt"}),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        store.add(embeddings, chunks)

        # Use a mock embedding engine that returns predictable vectors
        class MockEmbeddingEngine:
            dimension = 4
            def embed_query(self, query: str) -> np.ndarray:
                # Return a vector close to the first chunk
                return np.array([0.9, 0.05, 0.05, 0.0], dtype=np.float32)
            def embed_texts(self, texts, batch_size=64):
                return np.random.rand(len(texts), 4).astype(np.float32)

        retriever = Retriever(store, MockEmbeddingEngine())
        return QAChain(retriever)

    def test_extractive_answer(self, chain):
        """Extractive mode should return relevant passages."""
        result = chain.answer("What is AI?", use_llm=False)

        assert isinstance(result, Answer)
        assert result.mode == "extractive"
        assert len(result.passages) > 0
        assert result.question == "What is AI?"
        assert "ai.txt" in result.sources or len(result.sources) > 0

    def test_answer_includes_source_citations(self, chain):
        """Answer text should reference source files."""
        result = chain.answer("Tell me about AI", use_llm=False)
        # The extractive answer format includes source file names
        assert any(src in result.answer_text for src in ["ai.txt", "python.txt", "ml.txt"])

    def test_no_results_message(self):
        """Empty store should return a helpful no-results message."""
        store = VectorStore(dimension=4)

        class MockEngine:
            dimension = 4
            def embed_query(self, q):
                return np.array([1, 0, 0, 0], dtype=np.float32)

        retriever = Retriever(store, MockEngine())
        chain = QAChain(retriever)
        result = chain.answer("anything")

        assert "No relevant passages" in result.answer_text
        assert result.passages == []
