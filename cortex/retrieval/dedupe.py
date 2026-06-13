"""Deduplicate retrieved chunks by source path."""

from __future__ import annotations

from cortex.retrieval.kb_retriever import RetrievedChunk


def dedupe_by_path(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the best-scoring chunk per relative_path."""
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = chunk.relative_path
        existing = seen.get(key)
        if existing is None:
            seen[key] = chunk
            continue
        existing_score = (
            existing.rerank_score if existing.rerank_score is not None else existing.score
        )
        chunk_score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
        if chunk_score > existing_score:
            seen[key] = chunk

    order = [c.relative_path for c in chunks]
    unique_paths: list[str] = []
    for path in order:
        if path in seen and path not in unique_paths:
            unique_paths.append(path)
    return [seen[path] for path in unique_paths]
