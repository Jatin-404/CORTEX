"""Corrective-RAG context grader — relevance and sufficiency checks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from cortex.grading.prompts import build_grade_messages, build_rewrite_messages
from cortex.llm.ollama import OllamaChatClient
from cortex.retrieval.format import format_retrieval_results
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradeResult:
    relevant: bool
    sufficient: bool
    confidence: float
    reason: str
    method: str  # "fast_pass" | "fast_fail" | "llm"

    @property
    def passed(self) -> bool:
        return self.relevant and self.sufficient


@dataclass(frozen=True)
class GradeAttempt:
    query: str
    grade: GradeResult
    chunk_count: int


class ContextGrader:
    """Scores retrieved chunks before synthesis; rewrites query on failure."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = OllamaChatClient(self.settings)

    def grade(self, query: str, chunks: list[RetrievedChunk]) -> GradeResult:
        if not chunks:
            return GradeResult(
                relevant=False,
                sufficient=False,
                confidence=0.0,
                reason="No chunks retrieved",
                method="fast_fail",
            )

        top_score = _best_score(chunks)

        if top_score >= self.settings.grader_fast_pass_score:
            return GradeResult(
                relevant=True,
                sufficient=True,
                confidence=top_score,
                reason=f"Top rerank score {top_score:.3f} exceeds fast-pass threshold",
                method="fast_pass",
            )

        if top_score < self.settings.grader_min_rerank_score:
            return GradeResult(
                relevant=False,
                sufficient=False,
                confidence=top_score,
                reason=f"Top rerank score {top_score:.3f} below minimum threshold",
                method="fast_fail",
            )

        return self._llm_grade(query, chunks)

    def rewrite_query(
        self,
        original_query: str,
        current_query: str,
        grade: GradeResult,
    ) -> str:
        messages = build_rewrite_messages(original_query, current_query, grade.reason)
        rewritten = self.llm.chat(
            messages,
            model=self.settings.grader_model or None,
            temperature=0.2,
            trace_name="grader-rewrite",
        )
        cleaned = rewritten.strip().strip('"').strip("'")
        log.info("query_rewritten", extra={"from": current_query, "to": cleaned})
        return cleaned or current_query

    def _llm_grade(self, query: str, chunks: list[RetrievedChunk]) -> GradeResult:
        context = format_retrieval_results(
            chunks,
            include_parent=True,
            max_content_chars=self.settings.grader_context_chars,
        )
        messages = build_grade_messages(query, context)
        raw = self.llm.chat(
            messages,
            model=self.settings.grader_model or None,
            temperature=0.0,
            trace_name="grader-llm",
        )

        try:
            data = _parse_json_object(raw)
            relevant = bool(data.get("relevant", False))
            sufficient = bool(data.get("sufficient", False))
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "LLM grade"))
            return GradeResult(
                relevant=relevant,
                sufficient=sufficient,
                confidence=confidence,
                reason=reason,
                method="llm",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("grader_parse_failed", extra={"error": str(exc), "raw": raw[:200]})
            top = _best_score(chunks)
            passed = top >= self.settings.grader_fast_pass_score
            return GradeResult(
                relevant=passed,
                sufficient=passed,
                confidence=top,
                reason="LLM grade parse failed; fell back to rerank score",
                method="llm",
            )


def _best_score(chunks: list[RetrievedChunk]) -> float:
    scores = [
        c.rerank_score if c.rerank_score is not None else c.score
        for c in chunks
    ]
    return max(scores) if scores else 0.0


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))

    raise json.JSONDecodeError("No JSON object found", text, 0)
