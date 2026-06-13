"""BGE cross-encoder reranker — scores (query, chunk) pairs for relevance."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache

from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.settings import Settings

log = logging.getLogger(__name__)


def build_rerank_text(chunk: RetrievedChunk, *, max_chars: int) -> str:
    """Assemble title + heading + section body for cross-encoder input."""
    parts: list[str] = []
    if chunk.title:
        parts.append(chunk.title.strip())
    if chunk.heading_path:
        parts.append(chunk.heading_path.strip())
    body = (chunk.parent_content or chunk.content).strip()
    if body:
        parts.append(body)

    text = "\n".join(parts)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


class BGEReranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        top_k = top_k or self.settings.rerank_top_k
        pairs = [
            (query, build_rerank_text(c, max_chars=self.settings.rerank_max_chars))
            for c in chunks
        ]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for chunk, raw_score in ranked[:top_k]:
            results.append(replace(chunk, rerank_score=float(raw_score)))

        log.debug(
            "rerank_complete",
            extra={
                "candidates": len(chunks),
                "top_k": top_k,
                "top_score": results[0].rerank_score if results else None,
            },
        )
        return results

    @property
    def model(self):
        return _load_cross_encoder(self.settings.rerank_model)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    log.info("loading_reranker", extra={"model": model_name})
    return CrossEncoder(model_name)
