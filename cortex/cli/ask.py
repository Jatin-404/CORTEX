"""CLI for cited KB answers via retrieve -> rerank -> grade -> synthesize."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from cortex.logging_config import configure_logging
from cortex.models.enums import SourceType
from cortex.settings import settings
from cortex.synthesis.synthesizer import KBSynthesizer, SynthesisResult


def _safe_text(text: str) -> str:
    """Avoid Windows console encoding errors on handbook punctuation."""
    return text.encode("ascii", "replace").decode("ascii")


def _format_sources(result: SynthesisResult) -> str:
    if not result.sources:
        return ""
    lines = ["Sources:"]
    for src in result.sources:
        heading = f" - {_safe_text(src.heading_path)}" if src.heading_path else ""
        score = (
            f"rerank={src.rerank_score:.4f}"
            if src.rerank_score is not None
            else f"retrieval={src.retrieval_score:.4f}"
        )
        lines.append(
            f"  [{src.index}] {_safe_text(src.title)} ({src.relative_path})"
            f"{heading} [{score}]"
        )
    return "\n".join(lines)


def _format_grade_trace(result: SynthesisResult) -> str:
    if not result.grade_attempts:
        return ""
    lines = ["Grading:"]
    for i, attempt in enumerate(result.grade_attempts, start=1):
        status = "PASS" if attempt.grade.passed else "FAIL"
        lines.append(
            f"  attempt {i} [{status}] query={attempt.query!r} "
            f"chunks={attempt.chunk_count} method={attempt.grade.method} "
            f"reason={_safe_text(attempt.grade.reason)}"
        )
    if result.final_query and result.final_query != result.query:
        lines.append(f"  final_query={result.final_query!r}")
    return "\n".join(lines)


def run_ask(
    query: str,
    *,
    limit: int = 5,
    department: str | None = None,
    source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
    use_rerank: bool | None = None,
    use_grader: bool | None = None,
    show_sources: bool = True,
    show_grade: bool = False,
    as_json: bool = False,
) -> SynthesisResult:
    synthesizer = KBSynthesizer(settings)
    result = synthesizer.ask(
        query,
        source_type=source_type,
        limit=limit,
        department=department,
        use_rerank=use_rerank,
        use_grader=use_grader,
    )

    if as_json:
        print(
            json.dumps(
                {
                    "query": result.query,
                    "final_query": result.final_query,
                    "answer": result.answer,
                    "grade_passed": result.grade_passed,
                    "grade_attempts": [
                        {
                            "query": a.query,
                            "chunk_count": a.chunk_count,
                            "passed": a.grade.passed,
                            "relevant": a.grade.relevant,
                            "sufficient": a.grade.sufficient,
                            "confidence": a.grade.confidence,
                            "method": a.grade.method,
                            "reason": a.grade.reason,
                        }
                        for a in result.grade_attempts
                    ],
                    "sources": [
                        {
                            "index": s.index,
                            "title": s.title,
                            "relative_path": s.relative_path,
                            "heading_path": s.heading_path,
                            "rerank_score": s.rerank_score,
                            "retrieval_score": s.retrieval_score,
                        }
                        for s in result.sources
                    ],
                },
                indent=2,
            )
        )
        return result

    print(f'Question: "{result.query}"')
    print()
    print(result.answer)
    print()
    if show_grade and result.grade_attempts:
        print(_format_grade_trace(result))
        print()
    if show_sources and result.sources:
        print(_format_sources(result))

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Cortex knowledge base")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--department", help="Filter by handbook department folder")
    parser.add_argument(
        "--source-type",
        default=SourceType.HANDBOOK_MARKDOWN.value,
        choices=[st.value for st in SourceType],
    )
    parser.add_argument("--no-rerank", action="store_true", help="Skip BGE reranker")
    parser.add_argument("--no-grade", action="store_true", help="Skip Corrective-RAG grader")
    parser.add_argument("--show-grade", action="store_true", help="Print grading attempts")
    parser.add_argument("--no-sources", action="store_true", help="Hide source list after answer")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress httpx logs")

    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else settings.log_level)

    try:
        result = run_ask(
            args.query,
            limit=args.limit,
            department=args.department,
            source_type=SourceType(args.source_type),
            use_rerank=False if args.no_rerank else None,
            use_grader=False if args.no_grade else None,
            show_sources=not args.no_sources,
            show_grade=args.show_grade,
            as_json=args.json,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("ask_failed: %s", exc)
        return 1

    return 0 if result.answer else 2


if __name__ == "__main__":
    sys.exit(main())
