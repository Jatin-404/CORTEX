"""CLI for hybrid KB retrieval with optional BGE reranking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from cortex.logging_config import configure_logging
from cortex.models.enums import SourceType
from cortex.retrieval.kb_retriever import RetrievedChunk
from cortex.retrieval.pipeline import RetrievalPipeline
from cortex.settings import settings


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _score_label(hit: RetrievedChunk) -> str:
    if hit.rerank_score is not None:
        return f"rerank={hit.rerank_score:.4f} retrieval={hit.score:.4f}"
    return f"score={hit.score:.4f}"


def _format_text_result(rank: int, hit: RetrievedChunk, *, show_parent: bool, preview: int) -> str:
    lines = [
        f"[{rank}] {_score_label(hit)}",
        f"    source: {hit.relative_path}",
        f"    title:  {hit.title}",
    ]
    if hit.heading_path:
        lines.append(f"    heading: {hit.heading_path}")
    lines.append("    ---")
    lines.append(textwrap.indent(_truncate(hit.content, preview), "    "))
    if show_parent and hit.parent_content and hit.parent_content != hit.content:
        lines.append("    --- parent context ---")
        lines.append(textwrap.indent(_truncate(hit.parent_content, preview * 2), "    "))
    return "\n".join(lines)


def run_query(
    query: str,
    *,
    limit: int = 5,
    department: str | None = None,
    source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
    show_parent: bool = False,
    preview: int = 300,
    as_json: bool = False,
    use_rerank: bool | None = None,
) -> list[RetrievedChunk]:
    pipeline = RetrievalPipeline(settings)
    results = pipeline.search(
        query,
        source_type=source_type,
        limit=limit,
        department=department,
        use_rerank=use_rerank,
    )

    rerank_on = settings.rerank_enabled if use_rerank is None else use_rerank

    if as_json:
        payload = [
            {
                "rank": i,
                "retrieval_score": hit.score,
                "rerank_score": hit.rerank_score,
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "title": hit.title,
                "relative_path": hit.relative_path,
                "heading_path": hit.heading_path,
                "source_type": hit.source_type,
                "content": hit.content,
                "parent_content": hit.parent_content if show_parent else None,
            }
            for i, hit in enumerate(results, start=1)
        ]
        print(json.dumps(payload, indent=2))
        return results

    print(f'Query: "{query}"')
    print(
        f"Results: {len(results)} "
        f"(source_type={source_type.value}, limit={limit}, rerank={rerank_on})"
    )
    if rerank_on:
        print(f"Pipeline: hybrid top-{settings.retrieve_candidates} -> rerank -> top-{limit}")
    if department:
        print(f"Department filter: {department}")
    print()

    if not results:
        print("No results found.")
        return results

    for rank, hit in enumerate(results, start=1):
        print(_format_text_result(rank, hit, show_parent=show_parent, preview=preview))
        print()

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the Cortex knowledge base")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Final result count (default: 5)")
    parser.add_argument(
        "--department",
        help="Filter by handbook department folder (e.g. engineering, values)",
    )
    parser.add_argument(
        "--source-type",
        default=SourceType.HANDBOOK_MARKDOWN.value,
        choices=[st.value for st in SourceType],
        help="Qdrant payload source_type filter",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder rerank; return raw hybrid search results",
    )
    parser.add_argument(
        "--show-parent",
        action="store_true",
        help="Include parent section context in output",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=300,
        help="Max characters to show per text snippet",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress httpx logs")

    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else settings.log_level)

    try:
        results = run_query(
            args.query,
            limit=args.limit,
            department=args.department,
            source_type=SourceType(args.source_type),
            show_parent=args.show_parent,
            preview=args.preview,
            as_json=args.json,
            use_rerank=False if args.no_rerank else None,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("query_failed: %s", exc)
        return 1

    return 0 if results else 2


if __name__ == "__main__":
    sys.exit(main())
