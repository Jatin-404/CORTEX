"""Celery ingestion tasks."""

from __future__ import annotations

import logging

from cortex.ingestion.pipeline import IngestionPipeline
from cortex.worker.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(bind=True, name="cortex.ingest_handbook", max_retries=2)
def ingest_handbook_task(self, force: bool = False) -> dict:
    pipeline = IngestionPipeline()
    pipeline.initialize()
    result = pipeline.ingest_handbook(force=force)
    return {
        "job_id": result.job_id,
        "documents_processed": result.documents_processed,
        "documents_skipped": result.documents_skipped,
        "chunks_upserted": result.chunks_upserted,
        "error_count": len(result.errors),
    }


@celery_app.task(bind=True, name="cortex.ingest_file", max_retries=2)
def ingest_file_task(self, file_path: str, force: bool = True) -> dict:
    from pathlib import Path

    pipeline = IngestionPipeline()
    pipeline.initialize()
    chunks = pipeline.ingest_file(Path(file_path), force=force)
    return {"file_path": file_path, "chunks_upserted": chunks}
