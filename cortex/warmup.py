"""Startup warmup — load heavy models before serving queries."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from cortex.agent.runner import KBGraphAgent
from cortex.ingestion.embedding.ollama import OllamaEmbedder
from cortex.llm.ollama import OllamaChatClient
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.retrieval.reranker import BGEReranker
from cortex.settings import Settings
from cortex.storage.qdrant import QdrantVectorStore

log = logging.getLogger(__name__)


@dataclass
class WarmupStep:
    name: str
    duration_ms: int
    ok: bool
    detail: str = ""


@dataclass
class WarmupReport:
    ok: bool
    steps: list[WarmupStep] = field(default_factory=list)
    total_ms: int = 0


def _run_step(name: str, fn: Callable[[], None]) -> WarmupStep:
    start = time.perf_counter()
    try:
        fn()
        duration_ms = int((time.perf_counter() - start) * 1000)
        log.info("warmup_step_ok", extra={"step": name, "duration_ms": duration_ms})
        return WarmupStep(name=name, duration_ms=duration_ms, ok=True)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        log.error("warmup_step_failed", extra={"step": name, "error": str(exc)})
        return WarmupStep(name=name, duration_ms=duration_ms, ok=False, detail=str(exc))


def run_warmup(
    settings: Settings | None = None,
    *,
    skip_llm_ping: bool | None = None,
) -> WarmupReport:
    """Eagerly load Qdrant, embedders, reranker, Ollama chat, and the KB agent."""
    settings = settings or Settings()
    skip_llm = (
        settings.warmup_skip_llm_ping if skip_llm_ping is None else skip_llm_ping
    )
    steps: list[WarmupStep] = []
    start = time.perf_counter()

    def check_qdrant() -> None:
        store = QdrantVectorStore(settings)
        if not store.client.collection_exists(store.collection):
            raise RuntimeError(
                f"Qdrant collection {store.collection!r} not found at {settings.qdrant_url}"
            )

    def check_ollama() -> None:
        with httpx.Client(base_url=settings.ollama_base_url, timeout=10.0) as client:
            response = client.get("/api/tags")
            response.raise_for_status()

    def warmup_sparse() -> None:
        embedder = OllamaEmbedder(settings)
        embedder.embed_texts(["warmup"])

    def warmup_reranker() -> None:
        if not settings.rerank_enabled:
            return
        reranker = BGEReranker(settings)
        dummy = RetrievedChunk(
            chunk_id="warmup",
            doc_id="warmup",
            score=0.0,
            content="warmup chunk",
            parent_content="warmup parent",
            heading_path="",
            title="Warmup",
            relative_path="warmup.md",
            source_type="handbook_markdown",
            metadata={},
        )
        reranker.rerank("warmup query", [dummy], top_k=1)

    def warmup_llm() -> None:
        client = OllamaChatClient(settings)
        client.chat([{"role": "user", "content": "ping"}], trace_name="warmup-ping")

    def warmup_agent() -> None:
        KBGraphAgent(settings)

    step_fns: list[tuple[str, Callable[[], None]]] = [
        ("qdrant", check_qdrant),
        ("ollama", check_ollama),
        ("sparse_embed", warmup_sparse),
    ]
    if settings.rerank_enabled:
        step_fns.append(("reranker", warmup_reranker))
    if not skip_llm:
        step_fns.append(("llm", warmup_llm))
    step_fns.append(("agent", warmup_agent))

    for name, fn in step_fns:
        steps.append(_run_step(name, fn))
        if not steps[-1].ok:
            total_ms = int((time.perf_counter() - start) * 1000)
            return WarmupReport(ok=False, steps=steps, total_ms=total_ms)

    total_ms = int((time.perf_counter() - start) * 1000)
    log.info("warmup_complete", extra={"total_ms": total_ms})
    return WarmupReport(ok=True, steps=steps, total_ms=total_ms)
