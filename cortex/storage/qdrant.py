"""Qdrant vector store — single collection, source_type payload filtering."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from cortex.models.enums import SourceType
from cortex.settings import Settings

log = logging.getLogger(__name__)

_POINT_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class QdrantVectorStore:
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.client = QdrantClient(url=self.settings.qdrant_url)
        self.collection = self.settings.qdrant_collection

    def ensure_collection(self, dense_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            log.info("qdrant_collection_exists", extra={"collection": self.collection})
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.DENSE_VECTOR_NAME: qm.VectorParams(
                    size=dense_dim,
                    distance=qm.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                    modifier=qm.Modifier.IDF,
                ),
            },
        )

        self._ensure_payload_indexes()
        log.info("qdrant_collection_created", extra={"collection": self.collection})

    def _ensure_payload_indexes(self) -> None:
        keyword_fields = (
            "source_type",
            "doc_id",
            "chunk_id",
            "department",
            "subdepartment",
            "confidentiality_level",
            "relative_path",
        )
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )

        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="content",
            field_schema=qm.TextIndexParams(
                type=qm.TextIndexType.TEXT,
                tokenizer=qm.TokenizerType.WORD,
                min_token_len=2,
                lowercase=True,
            ),
        )

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        dense_vectors: list[list[float]],
        sparse_indices: list[list[int]],
        sparse_values: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> int:
        if not chunk_ids:
            return 0

        points = []
        for i, chunk_id in enumerate(chunk_ids):
            points.append(
                qm.PointStruct(
                    id=self._point_id(chunk_id),
                    vector={
                        self.DENSE_VECTOR_NAME: dense_vectors[i],
                        self.SPARSE_VECTOR_NAME: qm.SparseVector(
                            indices=sparse_indices[i],
                            values=sparse_values[i],
                        ),
                    },
                    payload=payloads[i],
                )
            )

        batch_size = self.settings.qdrant_upsert_batch
        upserted = 0
        for offset in range(0, len(points), batch_size):
            batch = points[offset : offset + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
            upserted += len(batch)

        return upserted

    def delete_by_doc_id(self, doc_id: str, source_type: SourceType) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="doc_id",
                            match=qm.MatchValue(value=doc_id),
                        ),
                        qm.FieldCondition(
                            key="source_type",
                            match=qm.MatchValue(value=source_type.value),
                        ),
                    ]
                )
            ),
        )

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse_indices: list[int],
        query_sparse_values: list[float],
        *,
        source_type: SourceType | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        extra_filters: list[qm.FieldCondition] | None = None,
    ) -> list[qm.ScoredPoint]:
        must: list[Any] = list(extra_filters or [])
        if source_type is not None:
            must.append(
                qm.FieldCondition(
                    key="source_type",
                    match=qm.MatchValue(value=source_type.value),
                )
            )

        query_filter = qm.Filter(must=must) if must else None

        return self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(
                    query=query_dense,
                    using=self.DENSE_VECTOR_NAME,
                    limit=limit * 4,
                    filter=query_filter,
                ),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=query_sparse_indices,
                        values=query_sparse_values,
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=limit * 4,
                    filter=query_filter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_POINT_NS, chunk_id))

    def build_chunk_payload(self, chunk_metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize chunk metadata into a Qdrant payload."""
        return {
            "chunk_id": chunk_metadata["chunk_id"],
            "doc_id": chunk_metadata["doc_id"],
            "chunk_index": chunk_metadata["chunk_index"],
            "source_type": chunk_metadata["source_type"],
            "title": chunk_metadata.get("title", ""),
            "content": chunk_metadata["content"],
            "parent_content": chunk_metadata.get("parent_content", ""),
            "heading_path": chunk_metadata.get("heading_path", ""),
            "department": chunk_metadata.get("department", ""),
            "subdepartment": chunk_metadata.get("subdepartment", ""),
            "confidentiality_level": chunk_metadata.get("confidentiality_level", "global"),
            "relative_path": chunk_metadata.get("relative_path", ""),
            "is_index": bool(chunk_metadata.get("is_index", False)),
            "last_updated": chunk_metadata.get("last_updated", ""),
            "file_hash": chunk_metadata.get("file_hash", ""),
        }
