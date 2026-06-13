"""Serialize synthesis results for API and eval output."""

from __future__ import annotations

from cortex.grading.grader import GradeAttempt
from cortex.observability.langfuse import trace_url
from cortex.settings import Settings
from cortex.synthesis.result import SourceCitation, SynthesisResult


def synthesis_result_to_dict(result: SynthesisResult, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    return {
        "query": result.query,
        "final_query": result.final_query,
        "answer": result.answer,
        "grade_passed": result.grade_passed,
        "grade_attempts": [_grade_attempt_to_dict(a) for a in result.grade_attempts],
        "sources": [_source_to_dict(s) for s in result.sources],
        "thread_id": result.thread_id,
        "trace_id": result.trace_id,
        "trace_url": trace_url(settings, result.trace_id) if result.trace_id else "",
    }


def _grade_attempt_to_dict(attempt: GradeAttempt) -> dict:
    return {
        "query": attempt.query,
        "chunk_count": attempt.chunk_count,
        "passed": attempt.grade.passed,
        "relevant": attempt.grade.relevant,
        "sufficient": attempt.grade.sufficient,
        "confidence": attempt.grade.confidence,
        "method": attempt.grade.method,
        "reason": attempt.grade.reason,
    }


def _source_to_dict(source: SourceCitation) -> dict:
    return {
        "index": source.index,
        "title": source.title,
        "relative_path": source.relative_path,
        "heading_path": source.heading_path,
        "rerank_score": source.rerank_score,
        "retrieval_score": source.retrieval_score,
    }
