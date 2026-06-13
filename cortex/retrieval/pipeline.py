"""
Two-stage retrieval: wide hybrid search → cross-encoder rerank.

Graph topology (Stage 2d): retrieve → rerank → grade → synthesize
Build order: 2a reranker first; grader plugs in between rerank and synthesize later.
"""

from __future__ import annotations

import logging

from cortex.models.enums import SourceType
from cortex.retrieval.kb_retriever import KBRetriever, RetrievedChunk
from cortex.retrieval.reranker import BGEReranker
from cortex.settings import Settings

log = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Orchestrates hybrid retrieval and optional BGE reranking.

    With rerank enabled (default):
      1. Fetch `retrieve_candidates` from Qdrant hybrid search
      2. Rerank with cross-encoder
      3. Return top `rerank_top_k` (or caller's `limit`)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.retriever = KBRetriever(self.settings)
        self._reranker: BGEReranker | None = None

    @property
    def reranker(self) -> BGEReranker:
        if self._reranker is None:
            self._reranker = BGEReranker(self.settings)
        return self._reranker

    def search(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int | None = None,
        department: str | None = None,
        confidentiality_level: str | None = None,
        use_rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        use_rerank = (
            self.settings.rerank_enabled if use_rerank is None else use_rerank
        )
        final_k = limit or (
            self.settings.rerank_top_k if use_rerank else self.settings.retrieve_candidates
        )

        candidate_k = self.settings.retrieve_candidates if use_rerank else final_k

        candidates = self.retriever.search(
            query,
            source_type=source_type,
            limit=candidate_k,
            department=department,
            confidentiality_level=confidentiality_level,
        )

        if not use_rerank or not candidates:
            return candidates[:final_k]

        log.info(
            "rerank_start",
            extra={"query_len": len(query), "candidates": len(candidates), "final_k": final_k},
        )
        return self.reranker.rerank(query, candidates, top_k=final_k)
