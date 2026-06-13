"""Unit tests for LangGraph KB agent routing and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from cortex.agent.deps import AgentDeps
from cortex.agent.graph import build_kb_graph
from cortex.agent.nodes import route_after_grade, route_after_retrieve
from cortex.agent.serialization import (
    chunk_to_dict,
    dict_to_chunk,
    dict_to_grade_attempt,
    grade_attempt_to_dict,
)
from cortex.grading.grader import ContextGrader, GradeAttempt, GradeResult
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.settings import Settings


def _chunk(path: str = "about/a.md", rerank: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="id-1",
        doc_id="doc-1",
        score=0.5,
        content="body",
        parent_content="parent",
        heading_path="Section",
        title="Title",
        relative_path=path,
        source_type="handbook_markdown",
        metadata={},
        rerank_score=rerank,
    )


def test_route_after_retrieve_with_grader() -> None:
    assert route_after_retrieve({"use_grader": True}) == "grade"


def test_route_after_retrieve_without_grader() -> None:
    assert route_after_retrieve({"use_grader": False}) == "synthesize"


def test_route_after_grade_pass() -> None:
    assert route_after_grade({"grade_passed": True}) == "synthesize"


def test_route_after_grade_fail_with_retries_left() -> None:
    state = {"grade_passed": False, "retry_count": 0, "max_retries": 2}
    assert route_after_grade(state) == "rewrite"
    state["retry_count"] = 1
    assert route_after_grade(state) == "rewrite"


def test_route_after_grade_fail_exhausted() -> None:
    state = {"grade_passed": False, "retry_count": 2, "max_retries": 2}
    assert route_after_grade(state) == "refuse"


def test_chunk_serialization_roundtrip() -> None:
    chunk = _chunk()
    restored = dict_to_chunk(chunk_to_dict(chunk))
    assert restored.chunk_id == chunk.chunk_id
    assert restored.relative_path == chunk.relative_path
    assert restored.rerank_score == chunk.rerank_score


def test_grade_attempt_serialization_roundtrip() -> None:
    attempt = GradeAttempt(
        query="q",
        chunk_count=2,
        grade=GradeResult(
            relevant=True,
            sufficient=True,
            confidence=0.9,
            reason="ok",
            method="fast_pass",
        ),
    )
    restored = dict_to_grade_attempt(grade_attempt_to_dict(attempt))
    assert restored.query == "q"
    assert restored.grade.passed


@dataclass
class _MockPipeline:
    chunks: list[RetrievedChunk]

    def search(self, query: str, **kwargs) -> list[RetrievedChunk]:
        return self.chunks


def test_graph_invoke_no_grader_synthesize_only() -> None:
    chunk = _chunk()
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "Answer with [1] citation."

    deps = AgentDeps(
        settings=Settings(grader_enabled=False, langgraph_checkpoint_backend="none"),
        pipeline=_MockPipeline([chunk]),
        grader=MagicMock(spec=ContextGrader),
        llm=mock_llm,
    )

    graph = build_kb_graph(deps=deps, checkpointer=None)
    config = {"configurable": {"deps": deps, "thread_id": "test-thread"}}

    final = graph.invoke(
        {
            "query": "What is CREDIT?",
            "current_query": "What is CREDIT?",
            "source_type": "handbook_markdown",
            "use_grader": False,
            "use_rerank": False,
            "limit": 5,
            "retry_count": 0,
            "max_retries": 2,
            "grade_attempts": [],
        },
        config,
    )

    assert "Answer with [1]" in final["answer"]
    assert final.get("grade_passed", True)
    assert len(final.get("chunks", [])) == 1
    mock_llm.chat.assert_called_once()
    deps.grader.grade.assert_not_called()


def test_graph_invoke_grader_fail_then_refuse() -> None:
    chunk = _chunk(rerank=0.1)
    mock_grader = MagicMock(spec=ContextGrader)
    fail_grade = GradeResult(
        relevant=False,
        sufficient=False,
        confidence=0.1,
        reason="not relevant",
        method="fast_fail",
    )
    mock_grader.grade.return_value = fail_grade
    mock_grader.rewrite_query.return_value = "rewritten query"

    deps = AgentDeps(
        settings=Settings(
            grader_enabled=True,
            grader_max_retries=1,
            langgraph_checkpoint_backend="none",
        ),
        pipeline=_MockPipeline([chunk]),
        grader=mock_grader,
        llm=MagicMock(),
    )

    graph = build_kb_graph(deps=deps, checkpointer=None)
    config = {"configurable": {"deps": deps, "thread_id": "test-fail"}}

    final = graph.invoke(
        {
            "query": "obscure topic",
            "current_query": "obscure topic",
            "source_type": "handbook_markdown",
            "use_grader": True,
            "use_rerank": False,
            "limit": 5,
            "retry_count": 0,
            "max_retries": 1,
            "grade_attempts": [],
        },
        config,
    )

    assert final.get("refused") is True
    assert "don't have enough information" in final["answer"].lower()
    assert mock_grader.grade.call_count == 2
    mock_grader.rewrite_query.assert_called_once()
    deps.llm.chat.assert_not_called()
