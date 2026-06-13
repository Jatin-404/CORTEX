"""KB retriever — hybrid search with source_type payload filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client.http import models as qm

from cortex.ingestion.embedding.ollama import OllamaEmbedder
from cortex.models.enums import SourceType
from cortex.settings import Settings
from cortex.storage.qdrant import QdrantVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    score: float
    content: str
    parent_content: str
    heading_path: str
    title: str
    relative_path: str
    source_type: str
    metadata: dict[str, Any]


class KBRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.embedder = OllamaEmbedder(self.settings)
        self.vector_store = QdrantVectorStore(self.settings)

    def search(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int = 10,
        department: str | None = None,
        confidentiality_level: str | None = None,
    ) -> list[RetrievedChunk]:
        embeddings = self.embedder.embed_texts([query])
        extra_filters: list[qm.FieldCondition] = []

        if department:
            extra_filters.append(
                qm.FieldCondition(
                    key="department",
                    match=qm.MatchValue(value=department),
                )
            )

        if confidentiality_level:
            extra_filters.append(
                qm.FieldCondition(
                    key="confidentiality_level",
                    match=qm.MatchValue(value=confidentiality_level),
                )
            )

        points = self.vector_store.hybrid_search(
            query_dense=embeddings.dense[0],
            query_sparse_indices=embeddings.sparse_indices[0],
            query_sparse_values=embeddings.sparse_values[0],
            source_type=source_type,
            limit=limit,
            extra_filters=extra_filters or None,
        )

        results: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    score=float(point.score or 0.0),
                    content=str(payload.get("content", "")),
                    parent_content=str(payload.get("parent_content", "")),
                    heading_path=str(payload.get("heading_path", "")),
                    title=str(payload.get("title", "")),
                    relative_path=str(payload.get("relative_path", "")),
                    source_type=str(payload.get("source_type", "")),
                    metadata=dict(payload),
                )
            )
        return results
