"""Postgres persistence for ingestion jobs and document change tracking."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from cortex.models.enums import SourceType
from cortex.settings import Settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class IngestionJobRow(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    documents_processed: Mapped[int] = mapped_column(Integer, default=0)
    documents_skipped: Mapped[int] = mapped_column(Integer, default=0)
    chunks_upserted: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)


class IngestedDocumentRow(Base):
    __tablename__ = "ingested_documents"

    doc_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IngestionJobStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.engine = create_engine(self.settings.postgres_dsn, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def start_job(
        self,
        job_type: str,
        source_type: SourceType,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        with self.SessionLocal() as session:
            session.add(
                IngestionJobRow(
                    id=job_id,
                    job_type=job_type,
                    source_type=source_type.value,
                    status="running",
                    started_at=_utcnow(),
                    metadata_json=metadata,
                )
            )
            session.commit()
        return job_id

    def complete_job(
        self,
        job_id: str,
        *,
        documents_processed: int,
        documents_skipped: int,
        chunks_upserted: int,
        error_message: str | None = None,
    ) -> None:
        status = "failed" if error_message else "completed"
        with self.SessionLocal() as session:
            row = session.get(IngestionJobRow, job_id)
            if row is None:
                return
            row.status = status
            row.completed_at = _utcnow()
            row.documents_processed = documents_processed
            row.documents_skipped = documents_skipped
            row.chunks_upserted = chunks_upserted
            row.error_message = error_message
            session.commit()


class DocumentRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.engine = create_engine(self.settings.postgres_dsn, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def get_hash(self, doc_id: str) -> str | None:
        with self.SessionLocal() as session:
            row = session.get(IngestedDocumentRow, doc_id)
            return row.file_hash if row else None

    def upsert(
        self,
        doc_id: str,
        source_type: SourceType,
        file_hash: str,
        relative_path: str,
        chunk_count: int,
    ) -> None:
        with self.SessionLocal() as session:
            row = session.get(IngestedDocumentRow, doc_id)
            if row is None:
                row = IngestedDocumentRow(
                    doc_id=doc_id,
                    source_type=source_type.value,
                    file_hash=file_hash,
                    relative_path=relative_path,
                    chunk_count=chunk_count,
                    last_ingested_at=_utcnow(),
                )
                session.add(row)
            else:
                row.source_type = source_type.value
                row.file_hash = file_hash
                row.relative_path = relative_path
                row.chunk_count = chunk_count
                row.last_ingested_at = _utcnow()
            session.commit()

    def list_doc_ids(self, source_type: SourceType) -> list[str]:
        with self.SessionLocal() as session:
            rows = session.scalars(
                select(IngestedDocumentRow.doc_id).where(
                    IngestedDocumentRow.source_type == source_type.value
                )
            ).all()
            return list(rows)
