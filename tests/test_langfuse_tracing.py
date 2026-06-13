"""Tests for Langfuse observability helpers."""

from __future__ import annotations

from cortex.observability.langfuse import is_enabled, summarize_chunks
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.settings import Settings


def _chunk(path: str = "about/a.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="id",
        doc_id="doc",
        score=0.5,
        content="body",
        parent_content="parent",
        heading_path="",
        title="Title",
        relative_path=path,
        source_type="handbook_markdown",
        metadata={},
        rerank_score=0.9,
    )


def test_is_enabled_false_by_default() -> None:
    assert not is_enabled(Settings(langfuse_enabled=False))


def test_is_enabled_requires_keys() -> None:
    assert not is_enabled(
        Settings(langfuse_enabled=True, langfuse_public_key="", langfuse_secret_key="x")
    )
    assert is_enabled(
        Settings(
            langfuse_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            langfuse_host="http://localhost:3000",
        )
    )


def test_summarize_chunks_omits_content() -> None:
    summary = summarize_chunks([_chunk()])
    assert summary[0]["relative_path"] == "about/a.md"
    assert "content" not in summary[0]
