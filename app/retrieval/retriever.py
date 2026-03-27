"""Retriever: similarity search with optional score filtering and reranking."""

from __future__ import annotations

import logging

from app.config import settings
from app.ingestion.embeddings import EmbeddingEngine
from app.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval interface over the vector store.

    Encapsulates embedding the query, performing similarity search,
    filtering by score threshold, and optionally reranking results.
    """

    def __init__(self, vector_store: VectorStore, embedding_engine: EmbeddingEngine | None = None) -> None:
        self.store = vector_store
        self.embeddings = embedding_engine or EmbeddingEngine()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a natural-language query.

        Args:
            query: The user's question.
            top_k: Max results to return.
            score_threshold: Minimum cosine similarity to include a result.

        Returns:
            List of result dicts sorted by descending relevance score.
            Each dict has keys: text, score, metadata.
        """
        top_k = top_k or settings.retrieval.top_k
        score_threshold = score_threshold or settings.retrieval.score_threshold

        query_embedding = self.embeddings.embed_query(query)
        results = self.store.search(query_embedding, top_k=top_k)

        # Filter by score threshold
        filtered = [r for r in results if r["score"] >= score_threshold]

        logger.info(
            "Query '%s' -> %d results (%d after threshold %.2f)",
            query[:60],
            len(results),
            len(filtered),
            score_threshold,
        )

        return filtered

    def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict:
        """Retrieve results and package them with summary statistics.

        Returns:
            Dict with keys: query, results, num_results, top_score, sources.
        """
        results = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)

        sources = list({r["metadata"].get("source", "unknown") for r in results})
        top_score = results[0]["score"] if results else 0.0

        return {
            "query": query,
            "results": results,
            "num_results": len(results),
            "top_score": top_score,
            "sources": sources,
        }
