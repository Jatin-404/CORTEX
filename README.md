# Cortex

Agentic retrieval platform — unified ingestion, hybrid vector search, and multi-source agent orchestration.

## Project layout

```
cortex/
├── cortex/
│   ├── models/          # Unified Document, Chunk, SourceType enums
│   ├── ingestion/
│   │   ├── parsers/     # HandbookParser (+ future PDF/DOCX)
│   │   ├── chunking/    # Parent-child semantic chunker
│   │   ├── embedding/   # Ollama dense + fastembed sparse (BM25)
│   │   └── pipeline.py  # Parse → chunk → embed → upsert
│   ├── storage/
│   │   ├── qdrant.py    # Single collection `cortex_vectors`
│   │   └── postgres.py  # Job logs + document hash registry
│   ├── retrieval/       # Hybrid search with source_type filters
│   ├── worker/          # Celery async ingestion
│   └── cli/             # `cortex-ingest` + `cortex-query` CLIs
├── db/migrations/       # SQL schema reference
├── tests/
├── docker-compose.yml
└── ARCHITECTURE.md
```

## Vector store design

One Qdrant collection (`cortex_vectors`) holds all embedded content. Payload field `source_type` discriminates:

| `source_type`        | Use case                          |
|----------------------|-----------------------------------|
| `handbook_markdown`  | GitLab handbook pages             |
| `semantic_layer`     | Schema/glossary for SQL agent     |
| `entity_notes`       | Versioned entity narrative fields |

Retrieval filters on `source_type` (+ `department`, `confidentiality_level`).

## Quick start

```bash
# 1. Start infrastructure
docker compose up -d postgres redis qdrant ollama

# 2. Pull embedding model
docker exec -it cortex-ollama-1 ollama pull nomic-embed-text

# 3. Install package
pip install -e ".[dev]"

# 4. Initialize + ingest handbook
cortex-ingest handbook --init-only   # creates tables + Qdrant collection
cortex-ingest handbook               # incremental (skips unchanged files)
cortex-ingest handbook --force       # full re-embed

# Single file
cortex-ingest file data/handbook-main/handbook-main/content/handbook/about/contributing.md
```

## Query the knowledge base

After ingestion, run hybrid search against Qdrant:

```bash
# Basic query
cortex-query "How do I contribute to the handbook?"

# More results + parent section context
cortex-query "GitLab CREDIT values" -n 5 --show-parent

# Filter to a department folder
cortex-query "deployment kubernetes" --department engineering

# JSON output (for scripting)
cortex-query "remote work policy" --json -q
```

Python API:

```python
from cortex.retrieval import KBRetriever, format_retrieval_results
from cortex.models.enums import SourceType

retriever = KBRetriever()
hits = retriever.search("GitLab values", source_type=SourceType.HANDBOOK_MARKDOWN, limit=5)
context = format_retrieval_results(hits)  # ready for LLM prompt
```

## Async ingestion (Celery)

```bash
docker compose up -d celery-worker

# From Python / API
from cortex.worker.tasks import ingest_handbook_task
ingest_handbook_task.delay(force=False)
```

## Configuration

Copy `.env.example` to `.env`. All settings are overridable via environment variables.
