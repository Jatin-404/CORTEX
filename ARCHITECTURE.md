# Agentic RAG Platform — System Architecture

## 1. Philosophy

This system is not "a chatbot with a vector database." It's an **agentic
retrieval platform** that reasons about *where* information lives, *how
fresh* it needs to be, *who is allowed to see it*, and *how confident* the
final answer is — before it ever talks to an LLM for synthesis.

Three data realities drive every design decision:

| Data type | Example | Nature | Primary retrieval strategy |
|---|---|---|---|
| Static knowledge base | Company policies, SOPs, wikis, manuals | Mostly immutable, document-shaped | Vector search (hybrid + rerank) |
| Transactional structured data | Orders, finance, project ledgers | Tabular, relational, exact-answer | Text-to-SQL over Postgres |
| Semi-dynamic entities | Land parcels, project status, acquisitions | Structured fields + evolving notes, versioned | Hybrid (structured filter + vector) + temporal queries |

A single **router/planner agent** decides which of these (often more than
one) a query needs, dispatches to specialized retriever agents, grades the
results, and synthesizes a cited answer — with every step traced, scored,
and stored for continuous evaluation.

---

## 2. High-Level Architecture

```
                                ┌─────────────────────────────────────────┐
                                │              CLIENT / API                │
                                │   (chat UI, REST/GraphQL gateway, auth)  │
                                └───────────────────┬───────────────────────┘
                                                    │  user_id, role, dept
                                                    ▼
                        ┌───────────────────────────────────────────────────┐
                        │                LANGGRAPH AGENT GRAPH                │
                        │                                                     │
                        │   ┌──────────┐   ┌────────────┐   ┌─────────────┐  │
                        │   │  Router/  │──▶│  Retriever  │──▶│   Grader /  │  │
                        │   │  Planner  │   │   Agents    │   │   Critic    │  │
                        │   └──────────┘   └────────────┘   └──────┬──────┘  │
                        │        │               │                  │ retry/  │
                        │        │         ┌─────┴─────┐            │ rewrite │
                        │        │         │           │            ▼         │
                        │        │   KB Vector   SQL Agent   ┌─────────────┐  │
                        │        │   Search      (Postgres)  │ Synthesizer │  │
                        │        │         │           │     │  + Citations│  │
                        │        │   Entity Hybrid     │     └──────┬──────┘  │
                        │        │   (parcels etc.)    │            │         │
                        │        │         └─────┬─────┘            ▼         │
                        │        │               │           ┌─────────────┐  │
                        │        └───────────────┴──────────▶│   Verifier  │  │
                        │                                     │ (hallucination│ │
                        │                                     │  check)      │  │
                        │                                     └──────┬──────┘  │
                        └────────────────────────────────────────────┼─────────┘
                                                                       ▼
                                                              Final answer + citations
                                                              + confidence score
```

Every box above emits structured trace events to **Langfuse**. Every box
also checks the caller's **access scope** (role/department) before touching
data.

---

## 3. Data Sources & Modeling

### 3.1 Static knowledge base
- Source formats: PDFs, DOCX, HTML exports, Confluence/Notion dumps.
- Parsed via `unstructured`, chunked semantically (by section/heading, not
  fixed character count), with **parent-child chunking**: small chunks
  (~300 tokens) for retrieval precision, larger parent sections (~1500
  tokens) returned to the LLM for context.
- Metadata attached to every chunk: `doc_id`, `title`, `department`,
  `confidentiality_level`, `version`, `last_updated`, `source_path`.

### 3.2 Transactional structured data
- Lives natively in Postgres (orders, finance ledgers, project budgets).
- **Never embedded as raw rows.** Instead, a "semantic layer" is built:
  table/column descriptions, business glossary terms, and a handful of
  example Q→SQL pairs are embedded in the vector DB.
- The SQL Agent retrieves the relevant schema context, generates SQL,
  validates it against an allow-list of read-only views (see §6 Access
  Control), executes it, and returns rows + a natural-language gloss.

### 3.3 Semi-dynamic entities (e.g., land parcels)
- Modeled as a Postgres table with structured fields (`parcel_id`,
  `status`, `location`, `acquisition_date`, `value`, `region`,
  `department_owner`) **plus** a free-text `notes`/`summary` field.
- On every update, a new row is appended to a `parcel_history` table
  (full audit trail), and the `notes` field is re-embedded with metadata
  `{parcel_id, version, valid_from, valid_to}`.
- This gives the agent two query modes on the *same* entity:
  - **Current state**: structured filter on the live table
    ("what's the status of parcel 114?")
  - **Temporal/historical**: vector + time-range filter on
    `parcel_history` ("how did parcel 114's status change over the last
    two quarters?")

---

## 4. Ingestion Pipeline

```
Source files / DB tables / entity updates
        │
        ▼
  Celery worker (Redis broker)
        │
        ├─ Hash check (skip if unchanged) ──▶ done
        │
        ├─ Parse + chunk (unstructured / custom parsers)
        │
        ├─ Embed (Ollama: nomic-embed-text / bge-large)
        │
        ├─ Tag metadata (dept, confidentiality, version, timestamps)
        │
        ├─ Upsert → Qdrant (vector + payload)
        │
        └─ Mirror structured data → Postgres (+ history table for
           versioned entities)
```

Key properties:
- **Async & incremental** — only changed content gets re-embedded,
  essential once you're at GB-scale.
- **Idempotent** — re-running ingestion never duplicates vectors (upsert
  on a deterministic chunk ID = hash of doc_id + chunk_index).
- **Observable** — every ingestion job logs to Postgres (`ingestion_jobs`
  table: status, duration, chunk counts, errors) and exposes Prometheus
  metrics (queue depth, throughput, failure rate).

---

## 5. Storage Stack

| Component | Role |
|---|---|
| **Qdrant** | Vector store for KB chunks, semantic-layer schema docs, and entity `notes`/history embeddings. Chosen for native hybrid (dense+BM25) search and strong payload filtering — critical for access control and temporal filters. |
| **Postgres** | System of record: transactional tables, entity tables + history, ingestion job logs, LangGraph checkpoints (conversation state), feedback table, eval results. |
| **Redis** | Celery broker for ingestion jobs; semantic cache for repeated/similar queries; short-term conversational memory; rate limiting. |
| **MinIO** | Raw document storage (source of truth for re-ingestion). |
| **Ollama** | Local model serving — embedding model + a small fast model (routing/classification/grading) + a larger model (final synthesis). |

---

## 6. Agentic Orchestration (LangGraph)

Nodes in the graph:

1. **Router/Planner** — classifies the query (KB / SQL / entity / multi-hop),
   decomposes complex questions into sub-questions, and attaches the
   caller's access scope to each sub-question.
2. **Retriever agents** (run in parallel where relevant):
   - *KB Retriever*: hybrid search + cross-encoder rerank (BGE reranker),
     filtered by `confidentiality_level` and `department`.
   - *SQL Agent*: schema-aware text-to-SQL, runs only against read-only
     views scoped to the caller's role.
   - *Entity Hybrid Retriever*: structured filter (current table) and/or
     vector + time-range filter (history table) depending on whether the
     question is about "now" or "then vs now."
3. **Grader/Critic (Corrective-RAG)** — scores retrieved context for
   relevance/sufficiency. If insufficient: rewrites the query and
   re-retrieves, or escalates to a different source. Caps retries (e.g., 2)
   to avoid infinite loops.
4. **Synthesizer** — combines all retrieved context into an answer with
   inline citations back to documents, SQL results, or parcel records
   (including version/date for temporal answers).
5. **Verifier** — lightweight self-check pass: does every claim in the
   draft answer trace back to retrieved context? Flags or strips
   unsupported claims, attaches a confidence score.

LangGraph checkpoints (Postgres-backed) persist state across turns, so
multi-turn conversations and "continue from here" replays work naturally —
and every checkpoint is a debuggable artifact in Langfuse.

---

## 7. Access Control & Data Scoping

A lightweight but real implementation, enterprise-flavored:

- Every user has a `role` and `department` (stored in Postgres, attached
  to the JWT/session at the API gateway).
- **Qdrant payload filters**: every vector search includes a mandatory
  filter on `confidentiality_level <= user.clearance` AND
  (`department == user.department` OR `department == "global"`).
- **SQL Agent**: generated SQL is validated against a per-role allow-list
  of views — e.g., a "finance analyst" role can query
  `vw_finance_summary` but not raw payroll tables. The agent never gets
  raw table access; it only sees views it's permitted to see in its
  schema context, so it can't even *generate* a query against
  forbidden tables.
- **Entity data**: parcel records carry a `department_owner` field used
  the same way as KB department tags.
- All access decisions are logged (who asked what, what scope was applied,
  what was returned) — this audit trail doubles as a security feature and
  a debugging tool.

This is intentionally simple (role + department, not a full RBAC/ABAC
engine) but demonstrates the *pattern* enterprises need: scoping happens
at the retrieval layer, not by trusting the LLM to "behave."

---

## 8. Versioned / Temporal Reasoning

This is the showcase feature for the land-acquisition use case.

- Every update to a tracked entity (e.g., a parcel) writes a new row to
  `parcel_history` with `valid_from`/`valid_to` timestamps — classic
  slowly-changing-dimension (SCD Type 2) modeling.
- The `notes` field of each historical version is embedded separately,
  tagged with its validity window.
- The Router detects temporal language ("last quarter," "how has X
  changed," "compare then vs now") and routes to the **Entity Hybrid
  Retriever in temporal mode**, which:
  1. Resolves the time range(s) referenced.
  2. Pulls the relevant historical row(s) + current row from Postgres.
  3. Pulls matching embedded `notes` from each version for narrative
     context.
  4. Hands the Synthesizer a structured diff (status A → status B,
     value change, date of change) plus narrative notes from each
     period.
- The answer can then say things like "As of Q1 2026, parcel 114 was
  'under negotiation' (per the Feb 2026 update); as of now it's
  'acquired,' finalized on May 3, 2026 — see [history v3]."

This is the part of the system that's genuinely hard to get right and is
a strong differentiator — most RAG demos can't reason about *change over
time* at all.

---

## 9. Feedback Loop & Continuous Evaluation

```
User answer ──▶ 👍 / 👎 (+ optional comment)
                    │
                    ▼
            Postgres `feedback` table
        (query, answer, trace_id, rating, comment, timestamp)
                    │
        ┌───────────┴────────────┐
        ▼                         ▼
  Curated eval set         Langfuse dataset
  (👎 + edge cases          (linked via trace_id
   reviewed → added         for full replay/debug)
   to regression suite)
                    │
                    ▼
          RAGAS evaluation run
   (faithfulness, relevance, context
    precision/recall) — scheduled,
    compared against baseline scores
                    │
                    ▼
       Dashboard: score trends over
       time as prompts/chunking/
       models change
```

- Every response is tagged with a `trace_id` linking it to its full
  Langfuse trace (router decision, retrieved chunks + scores, SQL
  generated, verifier output).
- Thumbs-down responses get triaged into a growing **regression eval
  set** — so the system has a measurable, growing "exam" it's graded
  against every time something changes (new prompt, new chunking
  strategy, new model).
- This turns the project from "I built a RAG demo" into "I built a RAG
  system with a measurable improvement loop" — which is exactly the kind
  of maturity an agentic-workflows team wants to see.

---

## 10. Observability Stack

| Tool | What it covers |
|---|---|
| **Langfuse** (self-hosted) | Full trace of every agent run: router decisions, tool calls, retrieved chunks + scores, SQL queries, token usage, latency, cost. Linked to feedback via `trace_id`. |
| **RAGAS** | Automated, scheduled evaluation of faithfulness, answer relevance, context precision/recall against the growing eval set. |
| **Prometheus + Grafana** | System metrics: ingestion throughput/queue depth, vector search latency, cache hit rate, Postgres query latency, error rates. |
| **Postgres tables** | `ingestion_jobs`, `feedback`, `access_logs`, `eval_runs` — the "ground truth" data behind all the dashboards above. |

The end result: for any answer, you can click through from the chat UI →
Langfuse trace → see exactly which retriever fired, what was returned,
why the critic accepted/rejected it, what the verifier flagged, and how
it scored on the latest eval run.

---

## 11. Tech Stack Summary

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM serving | Ollama (small model for routing/grading, larger model for synthesis) |
| Embeddings | Ollama (nomic-embed-text or bge-large) |
| Vector DB | Qdrant |
| Relational DB | Postgres (+ pgvector optional for the semantic layer if you want to keep it co-located) |
| Cache / queue | Redis + Celery |
| Object storage | MinIO |
| Reranking | BGE cross-encoder reranker |
| Observability | Langfuse, RAGAS, Prometheus, Grafana |
| Deployment | Docker Compose (local), structured so each service is independently scalable |

---

## 12. Roadmap / Suggested Build Order

1. **Foundations**: Docker Compose stack (Postgres, Redis, Qdrant, Ollama,
   MinIO) + basic ingestion for the static KB only.
2. **Single-source RAG**: hybrid search + rerank + simple LangGraph
   (retrieve → grade → synthesize). Add Langfuse tracing from day one.
3. **Add SQL Agent**: semantic layer + text-to-SQL over a small set of
   transactional tables, with access-scoped views.
4. **Add entity/temporal data**: parcel table + history table + temporal
   router branch.
5. **Access control**: wire role/department through the API gateway and
   into Qdrant filters + SQL view selection.
6. **Feedback loop + RAGAS eval set**: thumbs up/down, regression suite,
   scheduled eval runs.
7. **Polish observability dashboards** and write up the "click through a
   trace" demo — this is the part that sells the project.

Each stage is independently demoable, so you can show incremental
progress without needing the whole system finished.
