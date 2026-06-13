"""Dense (Ollama) + sparse (fastembed BM25) hybrid embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from fastembed import SparseTextEmbedding

from cortex.ingestion.embedding.base import EmbeddingBatch
from cortex.settings import Settings

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._sparse_model: SparseTextEmbedding | None = None

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name=self.settings.sparse_model)
        return self._sparse_model

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(dense=[], sparse_indices=[], sparse_values=[])

        dense = self._embed_dense(texts)
        sparse_indices, sparse_values = self._embed_sparse(texts)
        return EmbeddingBatch(
            dense=dense,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
        )

    def _embed_dense(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self.settings.embed_batch_size

        with httpx.Client(
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.embed_timeout_seconds,
        ) as client:
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset : offset + batch_size]
                for text in batch:
                    response = client.post(
                        "/api/embeddings",
                        json={"model": self.settings.embed_model, "prompt": text},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    vectors.append(payload["embedding"])

        return vectors

    def _embed_sparse(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        indices: list[list[int]] = []
        values: list[list[float]] = []

        for embedding in self.sparse_model.embed(texts):
            idx = embedding.indices.tolist()
            val = embedding.values.tolist()
            indices.append(idx)
            values.append(val)

        return indices, values
