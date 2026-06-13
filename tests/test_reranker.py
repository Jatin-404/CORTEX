"""Unit tests for reranker text building and ordering logic."""

from __future__ import annotations

from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.retrieval.reranker import build_rerank_text


def _chunk(**overrides) -> RetrievedChunk:
    defaults = {
        "chunk_id": "abc",
        "doc_id": "doc-1",
        "score": 0.5,
        "content": "child chunk text",
        "parent_content": "full parent section about contributing merge requests",
        "heading_path": "Contributing",
        "title": "Contributing to the Handbook",
        "relative_path": "about/contributing.md",
        "source_type": "handbook_markdown",
        "metadata": {},
    }
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_build_rerank_text_includes_title_heading_and_parent() -> None:
    text = build_rerank_text(_chunk(), max_chars=2000)
    assert "Contributing to the Handbook" in text
    assert "Contributing" in text
    assert "full parent section" in text


def test_build_rerank_text_truncates_long_body() -> None:
    long_body = "word " * 500
    text = build_rerank_text(_chunk(parent_content=long_body), max_chars=100)
    assert len(text) <= 100


def test_build_rerank_text_falls_back_to_content() -> None:
    text = build_rerank_text(_chunk(parent_content="", content="only child"), max_chars=500)
    assert "only child" in text
