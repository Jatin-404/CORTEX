"""Serialize retrieval/grade objects for LangGraph checkpoint state."""

from __future__ import annotations

from typing import Any

from cortex.grading.grader import GradeAttempt, GradeResult
from cortex.retrieval.kb_retriever import RetrievedChunk


def chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "score": chunk.score,
        "content": chunk.content,
        "parent_content": chunk.parent_content,
        "heading_path": chunk.heading_path,
        "title": chunk.title,
        "relative_path": chunk.relative_path,
        "source_type": chunk.source_type,
        "metadata": chunk.metadata,
        "rerank_score": chunk.rerank_score,
    }


def dict_to_chunk(data: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=data["chunk_id"],
        doc_id=data["doc_id"],
        score=float(data["score"]),
        content=data["content"],
        parent_content=data["parent_content"],
        heading_path=data.get("heading_path", ""),
        title=data.get("title", ""),
        relative_path=data["relative_path"],
        source_type=data.get("source_type", ""),
        metadata=dict(data.get("metadata", {})),
        rerank_score=data.get("rerank_score"),
    )


def grade_result_to_dict(grade: GradeResult) -> dict[str, Any]:
    return {
        "relevant": grade.relevant,
        "sufficient": grade.sufficient,
        "confidence": grade.confidence,
        "reason": grade.reason,
        "method": grade.method,
        "passed": grade.passed,
    }


def dict_to_grade_result(data: dict[str, Any]) -> GradeResult:
    return GradeResult(
        relevant=bool(data["relevant"]),
        sufficient=bool(data["sufficient"]),
        confidence=float(data["confidence"]),
        reason=str(data["reason"]),
        method=str(data["method"]),
    )


def grade_attempt_to_dict(attempt: GradeAttempt) -> dict[str, Any]:
    return {
        "query": attempt.query,
        "chunk_count": attempt.chunk_count,
        "grade": grade_result_to_dict(attempt.grade),
    }


def dict_to_grade_attempt(data: dict[str, Any]) -> GradeAttempt:
    return GradeAttempt(
        query=data["query"],
        chunk_count=int(data["chunk_count"]),
        grade=dict_to_grade_result(data["grade"]),
    )
