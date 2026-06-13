"""KB answer synthesis — retrieve, rerank, format context, generate cited answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cortex.llm.ollama import OllamaChatClient
from cortex.models.enums import SourceType
from cortex.retrieval.format import format_retrieval_results
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.retrieval.pipeline import RetrievalPipeline
from cortex.settings import Settings
from cortex.synthesis.prompts import build_messages

log = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    index: int
    title: str
    relative_path: str
    heading_path: str
    rerank_score: float | None
    retrieval_score: float


@dataclass
class SynthesisResult:
    query: str
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)


class KBSynthesizer:
    """
    End-to-end KB Q&A: RetrievalPipeline -> context formatting -> Ollama chat.

    LangGraph topology (later): rerank -> grade -> synthesize
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.pipeline = RetrievalPipeline(self.settings)
        self.llm = OllamaChatClient(self.settings)

    def ask(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int | None = None,
        department: str | None = None,
        use_rerank: bool | None = None,
    ) -> SynthesisResult:
        chunks = self._retrieve(
            query,
            source_type=source_type,
            limit=limit,
            department=department,
            use_rerank=use_rerank,
        )

        if not chunks:
            return SynthesisResult(
                query=query,
                answer="I don't have enough information in the handbook to answer that question.",
                sources=[],
                chunks=[],
            )

        context = format_retrieval_results(
            chunks,
            include_parent=True,
            max_content_chars=self.settings.synthesis_context_chars,
        )
        messages = build_messages(context, query)

        log.info(
            "synthesis_start",
            extra={"query_len": len(query), "chunks": len(chunks), "context_len": len(context)},
        )
        answer = self.llm.chat(messages)

        sources = [
            SourceCitation(
                index=i,
                title=hit.title,
                relative_path=hit.relative_path,
                heading_path=hit.heading_path,
                rerank_score=hit.rerank_score,
                retrieval_score=hit.score,
            )
            for i, hit in enumerate(chunks, start=1)
        ]

        return SynthesisResult(
            query=query,
            answer=answer,
            sources=sources,
            chunks=chunks,
        )

    def _retrieve(
        self,
        query: str,
        *,
        source_type: SourceType,
        limit: int | None,
        department: str | None,
        use_rerank: bool | None,
    ) -> list[RetrievedChunk]:
        raw = self.pipeline.search(
            query,
            source_type=source_type,
            limit=limit,
            department=department,
            use_rerank=use_rerank,
        )
        return _dedupe_by_path(raw)


def _dedupe_by_path(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the best-scoring chunk per relative_path to reduce duplicate doc noise."""
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = chunk.relative_path
        existing = seen.get(key)
        if existing is None:
            seen[key] = chunk
            continue
        existing_score = existing.rerank_score if existing.rerank_score is not None else existing.score
        chunk_score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
        if chunk_score > existing_score:
            seen[key] = chunk

    order = [c.relative_path for c in chunks]
    unique_paths: list[str] = []
    for path in order:
        if path in seen and path not in unique_paths:
            unique_paths.append(path)
    return [seen[path] for path in unique_paths]
