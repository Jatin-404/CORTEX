"""KB answer synthesis — retrieve, rerank, grade, synthesize (Corrective-RAG)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cortex.grading.grader import ContextGrader, GradeAttempt, GradeResult
from cortex.llm.ollama import OllamaChatClient
from cortex.models.enums import SourceType
from cortex.retrieval.format import format_retrieval_results
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.retrieval.pipeline import RetrievalPipeline
from cortex.settings import Settings
from cortex.synthesis.prompts import build_messages

log = logging.getLogger(__name__)

_INSUFFICIENT_MSG = (
    "I don't have enough information in the handbook to answer that question."
)


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
    final_query: str = ""
    grade_passed: bool = True
    grade_attempts: list[GradeAttempt] = field(default_factory=list)


class KBSynthesizer:
    """
    End-to-end KB Q&A with Corrective-RAG grading.

    LangGraph topology (later): retrieve -> rerank -> grade -> synthesize
    Retry loop: on grade failure, rewrite query and re-retrieve (capped).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.pipeline = RetrievalPipeline(self.settings)
        self.grader = ContextGrader(self.settings)
        self.llm = OllamaChatClient(self.settings)

    def ask(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int | None = None,
        department: str | None = None,
        use_rerank: bool | None = None,
        use_grader: bool | None = None,
    ) -> SynthesisResult:
        use_grader = self.settings.grader_enabled if use_grader is None else use_grader
        max_attempts = self.settings.grader_max_retries + 1

        current_query = query
        grade_attempts: list[GradeAttempt] = []
        chunks: list[RetrievedChunk] = []
        last_grade: GradeResult | None = None

        for attempt in range(max_attempts):
            chunks = self._retrieve(
                current_query,
                source_type=source_type,
                limit=limit,
                department=department,
                use_rerank=use_rerank,
            )

            if not use_grader:
                break

            grade = self.grader.grade(current_query, chunks)
            last_grade = grade
            grade_attempts.append(
                GradeAttempt(query=current_query, grade=grade, chunk_count=len(chunks))
            )

            log.info(
                "grade_result",
                extra={
                    "attempt": attempt + 1,
                    "passed": grade.passed,
                    "method": grade.method,
                    "reason": grade.reason,
                },
            )

            if grade.passed:
                break

            if attempt < max_attempts - 1:
                current_query = self.grader.rewrite_query(query, current_query, grade)
            else:
                return SynthesisResult(
                    query=query,
                    answer=_INSUFFICIENT_MSG,
                    sources=[],
                    chunks=chunks,
                    final_query=current_query,
                    grade_passed=False,
                    grade_attempts=grade_attempts,
                )

        if not chunks:
            return SynthesisResult(
                query=query,
                answer=_INSUFFICIENT_MSG,
                sources=[],
                chunks=[],
                final_query=current_query,
                grade_passed=False,
                grade_attempts=grade_attempts,
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
            final_query=current_query,
            grade_passed=last_grade.passed if last_grade else True,
            grade_attempts=grade_attempts,
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
