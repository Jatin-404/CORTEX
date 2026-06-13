"""Checkpoint backend selection for LangGraph."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import MemorySaver

from cortex.settings import Settings

log = logging.getLogger(__name__)


def normalize_postgres_dsn(dsn: str) -> str:
    """LangGraph PostgresSaver expects a plain postgresql:// URI."""
    return (
        dsn.replace("postgresql+psycopg://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )


@contextmanager
def get_checkpointer(settings: Settings | None = None) -> Iterator:
    settings = settings or Settings()
    backend = settings.langgraph_checkpoint_backend.lower()

    if backend == "none":
        yield None
        return

    if backend == "memory":
        yield MemorySaver()
        return

    # postgres (default)
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        conn = normalize_postgres_dsn(settings.postgres_dsn)
        with PostgresSaver.from_conn_string(conn) as checkpointer:
            checkpointer.setup()
            log.info("langgraph_checkpoint_postgres_ready")
            yield checkpointer
    except Exception as exc:
        log.warning(
            "langgraph_checkpoint_postgres_failed_fallback_memory",
            extra={"error": str(exc)},
        )
        yield MemorySaver()
