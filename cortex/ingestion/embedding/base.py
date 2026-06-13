from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingBatch:
    dense: list[list[float]]
    sparse_indices: list[list[int]]
    sparse_values: list[list[float]]


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        ...
