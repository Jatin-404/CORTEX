"""Integration tests for KBRetriever and rerank pipeline — require Qdrant + Ollama."""

from __future__ import annotations

import httpx
import pytest

from cortex.models.enums import SourceType
from cortex.retrieval.kb_retriever import KBRetriever
from cortex.retrieval.pipeline import RetrievalPipeline
from cortex.settings import Settings


def _services_available() -> bool:
    settings = Settings()
    try:
        with httpx.Client(timeout=3.0) as client:
            qdrant_ok = client.get(f"{settings.qdrant_url}/collections").is_success
            ollama_ok = client.get(f"{settings.ollama_base_url}/api/tags").is_success
        return qdrant_ok and ollama_ok
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _services_available(),
    reason="Qdrant and/or Ollama not reachable",
)


@pytest.fixture
def retriever() -> KBRetriever:
    return KBRetriever(Settings())


@pytest.fixture
def pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(Settings())


def test_search_returns_handbook_results(retriever: KBRetriever) -> None:
    results = retriever.search(
        "How do I contribute to the GitLab handbook?",
        source_type=SourceType.HANDBOOK_MARKDOWN,
        limit=5,
    )
    assert len(results) >= 1
    hit = results[0]
    assert hit.source_type == SourceType.HANDBOOK_MARKDOWN.value
    assert hit.content
    assert hit.relative_path
    assert hit.score > 0


def test_search_gitlab_values(retriever: KBRetriever) -> None:
    results = retriever.search(
        "GitLab CREDIT values collaboration transparency",
        source_type=SourceType.HANDBOOK_MARKDOWN,
        limit=3,
    )
    assert len(results) >= 1
    paths = {r.relative_path for r in results}
    assert any("values" in p for p in paths)


def test_department_filter(retriever: KBRetriever) -> None:
    results = retriever.search(
        "contributing merge requests",
        source_type=SourceType.HANDBOOK_MARKDOWN,
        department="about",
        limit=5,
    )
    assert len(results) >= 1
    assert all(r.metadata.get("department") == "about" for r in results)


def test_rerank_promotes_contributing_page(pipeline: RetrievalPipeline) -> None:
    query = "How do I contribute to the GitLab handbook?"
    hybrid = pipeline.search(query, limit=5, use_rerank=False)
    reranked = pipeline.search(query, limit=5, use_rerank=True)

    assert len(hybrid) >= 1
    assert len(reranked) >= 1
    assert all(hit.rerank_score is not None for hit in reranked)

    rerank_paths = [h.relative_path for h in reranked]
    assert any("contributing" in p for p in rerank_paths)

    # Reranker should surface the dedicated contributing page in top 2
    assert "about/contributing.md" in rerank_paths[:2]

