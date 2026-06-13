"""Invoke the LangGraph KB agent and map results to SynthesisResult."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langfuse import observe, propagate_attributes

from cortex.agent.checkpointer import get_checkpointer
from cortex.agent.deps import AgentDeps
from cortex.agent.graph import build_kb_graph
from cortex.agent.serialization import dict_to_chunk, dict_to_grade_attempt
from cortex.grading.grader import GradeAttempt
from cortex.models.enums import SourceType
from cortex.observability.langfuse import (
    configure,
    create_callback_handler,
    current_trace_id,
    flush,
    is_enabled,
    trace_url,
    update_current_span,
)
from cortex.settings import Settings
from cortex.synthesis.result import SourceCitation, SynthesisResult

log = logging.getLogger(__name__)


class KBGraphAgent:
    """LangGraph-backed KB Q&A agent with optional Postgres checkpoints."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.deps = AgentDeps.from_settings(self.settings)
        if is_enabled(self.settings):
            configure(self.settings)

    def ask(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int | None = None,
        department: str | None = None,
        use_rerank: bool | None = None,
        use_grader: bool | None = None,
        thread_id: str | None = None,
    ) -> SynthesisResult:
        if is_enabled(self.settings):
            return self._ask_traced(
                query,
                source_type=source_type,
                limit=limit,
                department=department,
                use_rerank=use_rerank,
                use_grader=use_grader,
                thread_id=thread_id,
            )
        return self._ask_impl(
            query,
            source_type=source_type,
            limit=limit,
            department=department,
            use_rerank=use_rerank,
            use_grader=use_grader,
            thread_id=thread_id,
        )

    @observe(name="cortex-kb-ask", capture_input=False, capture_output=False)
    def _ask_traced(
        self,
        query: str,
        *,
        source_type: SourceType,
        limit: int | None,
        department: str | None,
        use_rerank: bool | None,
        use_grader: bool | None,
        thread_id: str | None,
    ) -> SynthesisResult:
        thread_id = thread_id or str(uuid.uuid4())
        update_current_span(
            input={"query": query},
            metadata={
                "source_type": source_type.value,
                "department": department,
                "limit": limit,
                "use_rerank": use_rerank,
                "use_grader": use_grader,
            },
        )

        with propagate_attributes(
            session_id=thread_id,
            tags=["cortex", "kb-ask"],
            metadata={"feature": "handbook-rag"},
        ):
            result = self._ask_impl(
                query,
                source_type=source_type,
                limit=limit,
                department=department,
                use_rerank=use_rerank,
                use_grader=use_grader,
                thread_id=thread_id,
            )

        trace_id = current_trace_id()
        result.trace_id = trace_id
        update_current_span(
            output={
                "answer_preview": result.answer[:500],
                "grade_passed": result.grade_passed,
                "source_count": len(result.sources),
            },
            metadata={"trace_url": trace_url(self.settings, trace_id)},
        )
        flush(self.settings)
        return result

    def _ask_impl(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int | None = None,
        department: str | None = None,
        use_rerank: bool | None = None,
        use_grader: bool | None = None,
        thread_id: str | None = None,
    ) -> SynthesisResult:
        use_grader = self.settings.grader_enabled if use_grader is None else use_grader
        thread_id = thread_id or str(uuid.uuid4())

        initial_state: dict[str, Any] = {
            "query": query,
            "current_query": query,
            "source_type": source_type.value,
            "department": department,
            "limit": limit or self.settings.rerank_top_k,
            "use_rerank": self.settings.rerank_enabled if use_rerank is None else use_rerank,
            "use_grader": use_grader,
            "max_retries": self.settings.grader_max_retries,
            "retry_count": 0,
            "grade_attempts": [],
            "thread_id": thread_id,
        }

        config: dict[str, Any] = {
            "configurable": {
                "deps": self.deps,
                "thread_id": thread_id,
            }
        }

        handler = create_callback_handler()
        if handler is not None:
            config["callbacks"] = [handler]

        with get_checkpointer(self.settings) as checkpointer:
            graph = build_kb_graph(deps=self.deps, checkpointer=checkpointer)
            if checkpointer is not None:
                config["configurable"]["thread_id"] = thread_id
                final_state = graph.invoke(initial_state, config)
            else:
                final_state = graph.invoke(initial_state, config)

        return _state_to_result(query, final_state)

    def resume(self, thread_id: str, update: dict[str, Any] | None = None) -> SynthesisResult:
        """Resume a checkpointed thread (multi-turn: pass a new query in update)."""
        config: dict[str, Any] = {
            "configurable": {
                "deps": self.deps,
                "thread_id": thread_id,
            }
        }

        handler = create_callback_handler()
        if handler is not None:
            config["callbacks"] = [handler]

        with get_checkpointer(self.settings) as checkpointer:
            if checkpointer is None:
                raise RuntimeError("Cannot resume without a checkpointer backend")
            graph = build_kb_graph(deps=self.deps, checkpointer=checkpointer)
            final_state = graph.invoke(update or {}, config)
            query = final_state.get("query", "")
            result = _state_to_result(query, final_state)
            if is_enabled(self.settings):
                flush(self.settings)
            return result


def _state_to_result(query: str, state: dict[str, Any]) -> SynthesisResult:
    chunks = [dict_to_chunk(d) for d in state.get("chunks", [])]
    grade_attempts = [
        dict_to_grade_attempt(d) for d in state.get("grade_attempts", [])
    ]

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
        answer=state.get("answer", ""),
        sources=sources,
        chunks=chunks,
        final_query=state.get("final_query") or state.get("current_query") or query,
        grade_passed=bool(state.get("grade_passed", True)),
        grade_attempts=grade_attempts,
        thread_id=state.get("thread_id", ""),
    )
