"""Text chunking strategies for document splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with position and source metadata."""

    text: str
    index: int
    metadata: dict

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class RecursiveChunker:
    """Recursively splits text using a hierarchy of separators.

    Tries paragraph breaks first, then sentences, then words, falling back
    to character-level splits only when necessary. Maintains overlap between
    consecutive chunks for context continuity.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ):
        cfg = settings.chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.min_chunk_length = min_chunk_length or cfg.min_chunk_length

    def _split_text(self, text: str, separators: list[str] | None = None) -> list[str]:
        """Recursively split *text* into pieces no larger than *chunk_size*."""
        if separators is None:
            separators = list(self.SEPARATORS)

        final_chunks: list[str] = []

        # Find the best separator that actually exists in the text.
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        remaining_separators = separators[separators.index(separator) + 1 :] if separator in separators else []

        current_pieces: list[str] = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece) + (len(separator) if current_pieces else 0)

            if current_length + piece_len > self.chunk_size and current_pieces:
                merged = separator.join(current_pieces)
                if len(merged) > self.chunk_size and remaining_separators:
                    final_chunks.extend(self._split_text(merged, remaining_separators))
                else:
                    final_chunks.append(merged)

                # Keep overlap: retain trailing pieces that fit in the overlap budget.
                overlap_pieces: list[str] = []
                overlap_len = 0
                for p in reversed(current_pieces):
                    if overlap_len + len(p) + len(separator) > self.chunk_overlap:
                        break
                    overlap_pieces.insert(0, p)
                    overlap_len += len(p) + len(separator)

                current_pieces = overlap_pieces
                current_length = sum(len(p) for p in current_pieces) + len(separator) * max(0, len(current_pieces) - 1)

            current_pieces.append(piece)
            current_length += piece_len

        if current_pieces:
            merged = separator.join(current_pieces)
            if len(merged) > self.chunk_size and remaining_separators:
                final_chunks.extend(self._split_text(merged, remaining_separators))
            else:
                final_chunks.append(merged)

        return final_chunks

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split *text* into :class:`Chunk` objects.

        Args:
            text: The full document text.
            metadata: Base metadata to attach to every chunk (e.g. source).

        Returns:
            Ordered list of Chunk objects.
        """
        base_meta = metadata or {}
        raw_chunks = self._split_text(text)

        chunks: list[Chunk] = []
        for i, piece in enumerate(raw_chunks):
            piece = piece.strip()
            if len(piece) < self.min_chunk_length:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    index=i,
                    metadata={**base_meta, "chunk_index": i},
                )
            )

        logger.info(
            "Chunked document into %d chunks (size=%d, overlap=%d)",
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
