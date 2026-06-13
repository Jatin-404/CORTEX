"""Unit tests for ContextGrader fast-path and JSON parsing."""

from __future__ import annotations

from cortex.grading.grader import (
    ContextGrader,
    _best_score,
    _parse_json_object,
)
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.settings import Settings


def _chunk(path: str, score: float, rerank: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="id",
        doc_id="doc",
        score=score,
        content="body",
        parent_content="parent body",
        heading_path="Section",
        title="Title",
        relative_path=path,
        source_type="handbook_markdown",
        metadata={},
        rerank_score=rerank,
    )


def test_parse_json_object_plain() -> None:
    data = _parse_json_object(
        '{"relevant": true, "sufficient": false, "confidence": 0.5, "reason": "x"}'
    )
    assert data["relevant"] is True
    assert data["sufficient"] is False


def test_parse_json_object_fenced() -> None:
    data = _parse_json_object(
        '```json\n{"relevant": true, "sufficient": true, "confidence": 0.9, "reason": "ok"}\n```'
    )
    assert data["sufficient"] is True


def test_grade_fast_pass() -> None:
    grader = ContextGrader(Settings(grader_fast_pass_score=0.75))
    chunks = [_chunk("values/_index.md", 0.2, 0.99)]
    result = grader.grade("CREDIT values", chunks)
    assert result.passed
    assert result.method == "fast_pass"


def test_grade_fast_fail() -> None:
    grader = ContextGrader(Settings(grader_min_rerank_score=0.30))
    chunks = [_chunk("random.md", 0.1, 0.12)]
    result = grader.grade("obscure topic", chunks)
    assert not result.passed
    assert result.method == "fast_fail"


def test_grade_empty_chunks() -> None:
    grader = ContextGrader(Settings())
    result = grader.grade("anything", [])
    assert not result.passed
    assert result.reason == "No chunks retrieved"


def test_best_score_prefers_rerank() -> None:
    chunks = [_chunk("a.md", 0.5, 0.9), _chunk("b.md", 0.99, 0.2)]
    assert _best_score(chunks) == 0.9
