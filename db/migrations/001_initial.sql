-- Cortex initial schema (also created via SQLAlchemy on first run)

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              VARCHAR(36) PRIMARY KEY,
    job_type        VARCHAR(64) NOT NULL,
    source_type     VARCHAR(64) NOT NULL,
    status          VARCHAR(32) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    documents_processed INTEGER NOT NULL DEFAULT 0,
    documents_skipped   INTEGER NOT NULL DEFAULT 0,
    chunks_upserted     INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    metadata_json   JSONB
);

CREATE TABLE IF NOT EXISTS ingested_documents (
    doc_id          VARCHAR(36) PRIMARY KEY,
    source_type     VARCHAR(64) NOT NULL,
    file_hash       VARCHAR(64) NOT NULL,
    relative_path   VARCHAR(1024) NOT NULL,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    last_ingested_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ingested_documents_source_type
    ON ingested_documents (source_type);
