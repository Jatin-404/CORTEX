"""Format retrieval results for CLI and agent prompts."""

from __future__ import annotations

import textwrap

from cortex.retrieval.kb_retriever import RetrievedChunk


def format_retrieval_results(
    results: list[RetrievedChunk],
    *,
    include_parent: bool = True,
    max_content_chars: int = 800,
) -> str:
    """Build a single context block suitable for LLM synthesis."""
    if not results:
        return ""

    blocks: list[str] = []
    for i, hit in enumerate(results, start=1):
        citation = f"[{i}] {hit.title} ({hit.relative_path})"
        if hit.heading_path:
            citation += f" — {hit.heading_path}"

        body = hit.parent_content if include_parent and hit.parent_content else hit.content
        body = " ".join(body.split())
        if len(body) > max_content_chars:
            body = body[: max_content_chars - 3] + "..."

        blocks.append(f"{citation}\n{textwrap.indent(body, '  ')}")

    return "\n\n".join(blocks)
