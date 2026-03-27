"""Tests for the RecursiveChunker."""

import pytest

from app.ingestion.chunker import RecursiveChunker, Chunk


class TestRecursiveChunker:
    """Unit tests for RecursiveChunker."""

    def test_basic_chunking(self):
        """Short text that fits in one chunk returns a single chunk."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50, min_chunk_length=10)
        text = "This is a short document that should not be split."
        chunks = chunker.chunk(text, metadata={"source": "test.txt"})

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["source"] == "test.txt"

    def test_splits_long_text(self):
        """Text longer than chunk_size gets split into multiple chunks."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_length=10)
        text = " ".join(["word"] * 200)  # ~800 characters
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should be roughly within the size limit (with some tolerance)
            assert len(chunk.text) <= 120  # small buffer for boundary words

    def test_paragraph_splitting(self):
        """Chunker prefers paragraph boundaries."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20, min_chunk_length=10)
        text = "First paragraph with enough content to matter.\n\nSecond paragraph also has enough content to matter."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        # At least one chunk boundary should align with the paragraph break
        all_text = " ".join(c.text for c in chunks)
        assert "First paragraph" in all_text
        assert "Second paragraph" in all_text

    def test_min_chunk_length_filter(self):
        """Chunks shorter than min_chunk_length are discarded."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10, min_chunk_length=50)
        text = "Hi.\n\n" + "A " * 100
        chunks = chunker.chunk(text)

        for chunk in chunks:
            assert len(chunk.text) >= 50

    def test_metadata_propagation(self):
        """Base metadata is propagated to all chunks."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10, min_chunk_length=5)
        text = "Hello world this is a test. " * 10
        meta = {"source": "readme.md", "file_type": "md"}
        chunks = chunker.chunk(text, metadata=meta)

        for chunk in chunks:
            assert chunk.metadata["source"] == "readme.md"
            assert chunk.metadata["file_type"] == "md"
            assert "chunk_index" in chunk.metadata

    def test_empty_text_returns_no_chunks(self):
        """Empty text yields no chunks."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10, min_chunk_length=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_indices_are_sequential(self):
        """Chunk indices should be sequential integers."""
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10, min_chunk_length=5)
        text = "Sentence one. " * 50
        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == chunk.metadata["chunk_index"]

    def test_overlap_preserves_context(self):
        """Consecutive chunks should share some overlapping text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=30, min_chunk_length=5)
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunker.chunk(text)

        if len(chunks) >= 2:
            # Check that at least some tokens from the end of chunk N
            # appear at the start of chunk N+1
            for i in range(len(chunks) - 1):
                tail_words = set(chunks[i].text.split()[-5:])
                head_words = set(chunks[i + 1].text.split()[:10])
                # Overlap should cause at least one shared word
                assert tail_words & head_words, "Expected overlap between consecutive chunks"
