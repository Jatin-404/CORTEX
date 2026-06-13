"""Tests for startup warmup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cortex.settings import Settings
from cortex.warmup import run_warmup


def test_run_warmup_all_steps_ok() -> None:
    settings = Settings(rerank_enabled=True, warmup_skip_llm_ping=True)

    with (
        patch("cortex.warmup.QdrantVectorStore") as mock_store_cls,
        patch("cortex.warmup.httpx.Client") as mock_client_cls,
        patch("cortex.warmup.OllamaEmbedder") as mock_embedder_cls,
        patch("cortex.warmup.BGEReranker") as mock_reranker_cls,
        patch("cortex.warmup.KBGraphAgent") as mock_agent_cls,
    ):
        store = mock_store_cls.return_value
        store.client.collection_exists.return_value = True

        client = MagicMock()
        client.__enter__.return_value = client
        client.get.return_value.raise_for_status = MagicMock()
        mock_client_cls.return_value = client

        mock_embedder_cls.return_value.embed_texts.return_value = None
        mock_reranker_cls.return_value.rerank.return_value = []

        report = run_warmup(settings)

    assert report.ok
    step_names = [s.name for s in report.steps]
    assert "qdrant" in step_names
    assert "sparse_embed" in step_names
    assert "reranker" in step_names
    assert "llm" not in step_names
    assert "agent" in step_names
    mock_agent_cls.assert_called_once()


def test_run_warmup_fails_when_qdrant_missing() -> None:
    settings = Settings(warmup_skip_llm_ping=True)

    with patch("cortex.warmup.QdrantVectorStore") as mock_store_cls:
        store = mock_store_cls.return_value
        store.client.collection_exists.return_value = False

        report = run_warmup(settings)

    assert not report.ok
    assert report.steps[0].name == "qdrant"
    assert not report.steps[0].ok
