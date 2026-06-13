"""LangGraph node implementations for KB Q&A."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig

from cortex.agent.deps import get_deps
from cortex.agent.serialization import (
    chunk_to_dict,
    dict_to_chunk,
    dict_to_grade_attempt,
    grade_attempt_to_dict,
)
from cortex.agent.state import KBAgentState
from cortex.grading.grader import GradeAttempt
from cortex.models.enums import SourceType
from cortex.retrieval.dedupe import dedupe_by_path
from cortex.retrieval.format import format_retrieval_results
from cortex.synthesis.prompts import build_messages

log = logging.getLogger(__name__)

INSUFFICIENT_MSG = (
    "I don't have enough information in the handbook to answer that question."
)


def retrieve_rerank_node(state: KBAgentState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    current_query = state.get("current_query") or state["query"]
    source_type = SourceType(state.get("source_type", SourceType.HANDBOOK_MARKDOWN.value))

    raw = deps.pipeline.search(
        current_query,
        source_type=source_type,
        limit=state.get("limit"),
        department=state.get("department"),
        use_rerank=state.get("use_rerank"),
    )
    chunks = dedupe_by_path(raw)

    log.info("node_retrieve_rerank", extra={"query": current_query, "chunks": len(chunks)})
    return {
        "current_query": current_query,
        "chunks": [chunk_to_dict(c) for c in chunks],
    }


def grade_node(state: KBAgentState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    current_query = state.get("current_query") or state["query"]
    chunks = [dict_to_chunk(d) for d in state.get("chunks", [])]

    grade = deps.grader.grade(current_query, chunks)
    attempt = grade_attempt_to_dict(
        GradeAttempt(query=current_query, grade=grade, chunk_count=len(chunks))
    )

    log.info(
        "node_grade",
        extra={"passed": grade.passed, "method": grade.method, "reason": grade.reason},
    )
    return {
        "grade_passed": grade.passed,
        "grade_attempts": [attempt],
    }


def rewrite_node(state: KBAgentState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    attempts = state.get("grade_attempts", [])
    if not attempts:
        return {"retry_count": state.get("retry_count", 0) + 1}

    last = dict_to_grade_attempt(attempts[-1])
    current_query = state.get("current_query") or state["query"]
    new_query = deps.grader.rewrite_query(state["query"], current_query, last.grade)

    log.info("node_rewrite", extra={"from": current_query, "to": new_query})
    return {
        "current_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def synthesize_node(state: KBAgentState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    chunks = [dict_to_chunk(d) for d in state.get("chunks", [])]
    query = state["query"]
    current_query = state.get("current_query") or query

    if not chunks:
        return {
            "answer": INSUFFICIENT_MSG,
            "final_query": current_query,
            "grade_passed": False,
            "refused": True,
        }

    context = format_retrieval_results(
        chunks,
        include_parent=True,
        max_content_chars=deps.settings.synthesis_context_chars,
    )
    messages = build_messages(context, query)
    answer = deps.llm.chat(messages)

    log.info("node_synthesize", extra={"answer_len": len(answer), "chunks": len(chunks)})
    return {
        "answer": answer,
        "final_query": current_query,
        "refused": False,
    }


def refuse_node(state: KBAgentState) -> dict:
    current_query = state.get("current_query") or state["query"]
    log.info("node_refuse", extra={"query": current_query})
    return {
        "answer": INSUFFICIENT_MSG,
        "final_query": current_query,
        "grade_passed": False,
        "refused": True,
    }


def route_after_retrieve(state: KBAgentState) -> Literal["grade", "synthesize"]:
    if state.get("use_grader", True):
        return "grade"
    return "synthesize"


def route_after_grade(state: KBAgentState) -> Literal["synthesize", "rewrite", "refuse"]:
    if state.get("grade_passed"):
        return "synthesize"
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retry_count < max_retries:
        return "rewrite"
    return "refuse"
