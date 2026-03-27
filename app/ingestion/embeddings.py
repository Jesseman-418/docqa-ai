"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model (cached singleton)."""
    cfg = settings.embedding
    logger.info("Loading embedding model: %s (device=%s)", cfg.model_name, cfg.device)
    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    return model


class EmbeddingEngine:
    """Generate dense vector embeddings for text using sentence-transformers."""

    def __init__(self) -> None:
        self.model = _get_model()
        self.dimension = settings.embedding.dimension

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Args:
            texts: Strings to embed.
            batch_size: Batch size for encoding.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info("Embedded %d texts -> shape %s", len(texts), embeddings.shape)
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Returns:
            1-D numpy array of shape (dimension,).
        """
        return self.embed_texts([query])[0]
