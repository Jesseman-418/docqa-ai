"""Application configuration and settings."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model settings."""

    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"


@dataclass(frozen=True)
class ChunkingConfig:
    """Text chunking settings."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval settings."""

    top_k: int = 5
    score_threshold: float = 0.3


@dataclass
class AppConfig:
    """Top-level application configuration."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    index_dir: Path = Path("vector_store")
    supported_extensions: tuple[str, ...] = (".pdf", ".txt", ".md")

    # Optional LLM settings (only used when API key is provided)
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.0


# Singleton config instance
settings = AppConfig()
