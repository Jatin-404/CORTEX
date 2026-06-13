"""Central configuration — override via environment variables or .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Data sources ─────────────────────────────────────────────────────────
    handbook_root: Path = Path(
        "data/handbook-main/handbook-main/content/handbook"
    )

    # ── Embedding (Ollama) ───────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    embed_batch_size: int = 32
    embed_timeout_seconds: float = 120.0

    # ── Sparse vectors (fastembed BM25 — hybrid search) ──────────────────────
    sparse_model: str = "Qdrant/bm25"
    sparse_dim: int = 0  # auto-detected at runtime; 0 = defer to model

    # ── Chunking ─────────────────────────────────────────────────────────────
    child_chunk_tokens: int = 300
    parent_chunk_tokens: int = 1500
    chunk_overlap_tokens: int = 50

    # ── Qdrant (single collection, source_type payload filter) ─────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "cortex_vectors"
    qdrant_upsert_batch: int = 64

    # ── Postgres ─────────────────────────────────────────────────────────────
    postgres_dsn: str = "postgresql+psycopg://cortex:cortex@localhost:5432/cortex"

    # ── Redis / Celery ───────────────────────────────────────────────────────
    celery_broker: str = "redis://localhost:6379/0"
    celery_backend: str = "redis://localhost:6379/1"
    celery_task_default_queue: str = "ingestion"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Ingestion behaviour ───────────────────────────────────────────────────
    skip_unchanged: bool = True
    delete_stale_chunks: bool = True  # remove old chunks when doc changes


settings = Settings()
