"""Orchestrates parse → chunk → embed → upsert with change detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from cortex.ingestion.chunking.parent_child import ParentChildChunker
from cortex.ingestion.embedding.ollama import OllamaEmbedder
from cortex.ingestion.parsers.handbook import HandbookParser
from cortex.models.document import Chunk, Document
from cortex.models.enums import SourceType
from cortex.settings import Settings
from cortex.storage.postgres import DocumentRegistry, IngestionJobStore
from cortex.storage.qdrant import QdrantVectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    job_id: str
    documents_processed: int = 0
    documents_skipped: int = 0
    chunks_upserted: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.chunker = ParentChildChunker(
            child_chunk_tokens=self.settings.child_chunk_tokens,
            parent_chunk_tokens=self.settings.parent_chunk_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        self.embedder = OllamaEmbedder(self.settings)
        self.vector_store = QdrantVectorStore(self.settings)
        self.registry = DocumentRegistry(self.settings)
        self.job_store = IngestionJobStore(self.settings)

    def initialize(self) -> None:
        self.job_store.create_tables()
        self.vector_store.ensure_collection(self.settings.embed_dim)

    def ingest_handbook(self, *, force: bool = False) -> IngestionResult:
        handbook_root = self.settings.handbook_root
        if not handbook_root.exists():
            raise FileNotFoundError(f"Handbook root not found: {handbook_root}")

        parser = HandbookParser(handbook_root)
        job_id = self.job_store.start_job(
            job_type="full_handbook",
            source_type=SourceType.HANDBOOK_MARKDOWN,
            metadata={"handbook_root": str(handbook_root), "force": force},
        )

        result = IngestionResult(job_id=job_id)
        try:
            for document in parser.iter_documents():
                try:
                    upserted, skipped = self._ingest_document(document, force=force)
                    if skipped:
                        result.documents_skipped += 1
                    else:
                        result.documents_processed += 1
                        result.chunks_upserted += upserted
                except Exception as exc:
                    msg = f"{document.relative_path}: {exc}"
                    log.exception("document_ingest_failed", extra={"path": document.relative_path})
                    result.errors.append(msg)

            self.job_store.complete_job(
                job_id,
                documents_processed=result.documents_processed,
                documents_skipped=result.documents_skipped,
                chunks_upserted=result.chunks_upserted,
                error_message="; ".join(result.errors[:5]) if result.errors else None,
            )
        except Exception as exc:
            self.job_store.complete_job(
                job_id,
                documents_processed=result.documents_processed,
                documents_skipped=result.documents_skipped,
                chunks_upserted=result.chunks_upserted,
                error_message=str(exc),
            )
            raise

        log.info(
            "handbook_ingestion_complete",
            extra={
                "job_id": job_id,
                "processed": result.documents_processed,
                "skipped": result.documents_skipped,
                "chunks": result.chunks_upserted,
                "errors": len(result.errors),
            },
        )
        return result

    def ingest_file(self, path: Path, *, force: bool = True) -> int:
        parser = HandbookParser(self.settings.handbook_root)
        document = parser.parse_file(path)
        if document is None:
            raise ValueError(f"Could not parse: {path}")
        upserted, _ = self._ingest_document(document, force=force)
        return upserted

    def _ingest_document(self, document: Document, *, force: bool) -> tuple[int, bool]:
        if (
            self.settings.skip_unchanged
            and not force
            and self.registry.get_hash(document.doc_id) == document.file_hash
        ):
            log.debug("document_unchanged_skip", extra={"doc_id": document.doc_id})
            return 0, True

        chunks = self.chunker.chunk_document(document)
        if not chunks:
            log.warning("document_no_chunks", extra={"doc_id": document.doc_id})
            return 0, False

        if self.settings.delete_stale_chunks:
            self.vector_store.delete_by_doc_id(document.doc_id, document.source_type)

        upserted = self._upsert_chunks(chunks)

        self.registry.upsert(
            doc_id=document.doc_id,
            source_type=document.source_type,
            file_hash=document.file_hash,
            relative_path=document.relative_path,
            chunk_count=len(chunks),
        )
        return upserted, False

    def _upsert_chunks(self, chunks: list[Chunk]) -> int:
        total = 0
        batch_size = self.settings.embed_batch_size

        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedder.embed_texts(texts)

            chunk_ids: list[str] = []
            payloads: list[dict] = []

            for chunk in batch:
                chunk_ids.append(chunk.chunk_id)
                payload = self.vector_store.build_chunk_payload(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "source_type": chunk.source_type.value,
                        "content": chunk.content,
                        "parent_content": chunk.parent_content,
                        "heading_path": chunk.heading_path,
                        **chunk.metadata,
                    }
                )
                payloads.append(payload)

            total += self.vector_store.upsert_chunks(
                chunk_ids=chunk_ids,
                dense_vectors=embeddings.dense,
                sparse_indices=embeddings.sparse_indices,
                sparse_values=embeddings.sparse_values,
                payloads=payloads,
            )

        return total
