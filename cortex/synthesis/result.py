"""Synthesis result types — shared by synthesizer and LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from cortex.grading.grader import GradeAttempt
from cortex.retrieval.kb_retriever import RetrievedChunk


@dataclass
class SourceCitation:
    index: int
    title: str
    relative_path: str
    heading_path: str
    rerank_score: float | None
    retrieval_score: float


@dataclass
class SynthesisResult:
    query: str
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    final_query: str = ""
    grade_passed: bool = True
    grade_attempts: list[GradeAttempt] = field(default_factory=list)
    thread_id: str = ""
