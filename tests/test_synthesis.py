"""Unit tests for synthesis prompt building and deduplication."""

from __future__ import annotations

from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.synthesis.prompts import SYSTEM_PROMPT, build_messages
from cortex.synthesis.synthesizer import _dedupe_by_path


def _chunk(path: str, score: float, rerank: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"id-{path}",
        doc_id="doc",
        score=score,
        content=f"content for {path}",
        parent_content=f"parent for {path}",
        heading_path="",
        title=path,
        relative_path=path,
        source_type="handbook_markdown",
        metadata={},
        rerank_score=rerank,
    )


def test_build_messages_includes_context_and_query() -> None:
    messages = build_messages("context block", "What is CREDIT?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]
    assert "context block" in messages[1]["content"]
    assert "What is CREDIT?" in messages[1]["content"]


def test_dedupe_keeps_best_scoring_chunk_per_path() -> None:
    chunks = [
        _chunk("about/a.md", 0.5, 0.8),
        _chunk("about/a.md", 1.0, 0.6),
        _chunk("about/b.md", 0.3, 0.9),
    ]
    deduped = _dedupe_by_path(chunks)
    assert len(deduped) == 2
    assert deduped[0].relative_path == "about/a.md"
    assert deduped[0].rerank_score == 0.8
    assert deduped[1].relative_path == "about/b.md"
