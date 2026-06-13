"""Langfuse tracing helpers for Cortex KB agent."""

from __future__ import annotations

import os
from typing import Any

from cortex.settings import Settings

_langfuse_configured = False


def is_enabled(settings: Settings | None = None) -> bool:
    settings = settings or Settings()
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_host
    )


def configure(settings: Settings | None = None) -> None:
    """Set Langfuse env vars before client/handler initialization."""
    global _langfuse_configured
    settings = settings or Settings()
    if not is_enabled(settings):
        return

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host.rstrip("/")
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host.rstrip("/"))
    _langfuse_configured = True


def create_callback_handler():
    """LangGraph/LangChain callback handler; None when tracing is disabled."""
    settings = Settings()
    if not is_enabled(settings):
        return None

    configure(settings)
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def flush(settings: Settings | None = None) -> None:
    if not is_enabled(settings):
        return
    configure(settings)
    from langfuse import get_client

    get_client().flush()


def current_trace_id() -> str:
    """Return the active OpenTelemetry trace id as a 32-char hex string."""
    from opentelemetry import trace

    span = trace.get_current_span()
    if span is None:
        return ""
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return ""
    return format(ctx.trace_id, "032x")


def trace_url(settings: Settings | None = None, trace_id: str = "") -> str:
    if not trace_id:
        return ""
    settings = settings or Settings()
    base = settings.langfuse_host.rstrip("/")
    return f"{base}/trace/{trace_id}"


def update_current_span(*, input: Any = None, output: Any = None, metadata: Any = None) -> None:
    if not is_enabled():
        return
    configure()
    from langfuse import get_client

    get_client().update_current_span(input=input, output=output, metadata=metadata)


def update_current_generation(
    *,
    model: str | None = None,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
) -> None:
    if not is_enabled():
        return
    configure()
    from langfuse import get_client

    get_client().update_current_generation(
        model=model,
        input=input,
        output=output,
        metadata=metadata,
    )


def summarize_chunks(chunks: list[Any]) -> list[dict[str, Any]]:
    """Compact retrieval summary for trace output (no full chunk bodies)."""
    summary: list[dict[str, Any]] = []
    for chunk in chunks[:10]:
        summary.append(
            {
                "relative_path": getattr(chunk, "relative_path", ""),
                "title": getattr(chunk, "title", ""),
                "rerank_score": getattr(chunk, "rerank_score", None),
                "retrieval_score": getattr(chunk, "score", None),
            }
        )
    return summary


def record_retrieval(query: str, chunks: list[Any]) -> None:
    update_current_span(
        input={"query": query},
        output={"chunk_count": len(chunks), "chunks": summarize_chunks(chunks)},
    )


def record_grade(
    *,
    query: str,
    passed: bool,
    method: str,
    reason: str,
    chunk_count: int,
) -> None:
    update_current_span(
        input={"query": query},
        output={
            "passed": passed,
            "method": method,
            "reason": reason,
            "chunk_count": chunk_count,
        },
    )


def record_rewrite(*, from_query: str, to_query: str, retry_count: int) -> None:
    update_current_span(
        input={"from_query": from_query},
        output={"to_query": to_query, "retry_count": retry_count},
    )
