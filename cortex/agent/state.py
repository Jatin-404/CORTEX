"""LangGraph state schema for the KB agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class KBAgentState(TypedDict, total=False):
    # Input
    query: str
    current_query: str
    source_type: str
    department: str | None
    limit: int
    use_rerank: bool
    use_grader: bool
    max_retries: int

    # Retrieval + grading
    chunks: list[dict[str, Any]]
    grade_attempts: Annotated[list[dict[str, Any]], operator.add]
    grade_passed: bool
    retry_count: int

    # Output
    answer: str
    final_query: str
    refused: bool
    thread_id: str
