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
    postgres_dsn: str = "postgresql+psycopg://cortex:cortex@localhost:5433/cortex"

    # ── Redis / Celery ───────────────────────────────────────────────────────
    celery_broker: str = "redis://localhost:6379/0"
    celery_backend: str = "redis://localhost:6379/1"
    celery_task_default_queue: str = "ingestion"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Retrieval / reranking ─────────────────────────────────────────────────
    rerank_enabled: bool = True
    retrieve_candidates: int = 20  # wide pool from hybrid search before rerank
    rerank_top_k: int = 5          # final results after cross-encoder
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_chars: int = 1500   # max text length per cross-encoder pair

    # ── LLM synthesis (Stage 2b) ────────────────────────────────────────────
    llm_model: str = "llama3.2:latest"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 180.0
    synthesis_context_chars: int = 1200  # max chars per chunk in LLM context

    # ── Grader / Corrective-RAG (Stage 2c) ──────────────────────────────────
    grader_enabled: bool = True
    grader_max_retries: int = 2
    grader_model: str = ""  # empty = use llm_model
    grader_fast_pass_score: float = 0.75  # skip LLM grade when top rerank >= this
    grader_min_rerank_score: float = 0.30  # fast-fail when top rerank below this
    grader_context_chars: int = 800  # excerpt length sent to grader LLM

    # ── LangGraph (Stage 2d) ─────────────────────────────────────────────────
    langgraph_checkpoint_backend: str = "postgres"  # postgres | memory | none

    # ── Langfuse observability (Stage 2e) ────────────────────────────────────
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ── Ingestion behaviour ───────────────────────────────────────────────────
    skip_unchanged: bool = True
    delete_stale_chunks: bool = True  # remove old chunks when doc changes


settings = Settings()
