"""Tests for FastAPI service."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from cortex.api.app import create_app
from cortex.settings import Settings
from cortex.synthesis.result import SynthesisResult


def test_health_endpoint() -> None:
    app = create_app(Settings(warmup_enabled=False))
    with TestClient(app) as client:
        response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_without_warmup() -> None:
    app = create_app(Settings(warmup_enabled=False))
    with TestClient(app) as client:
        response = client.get("/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["warmup_ms"] == 0


def test_ask_endpoint_returns_serialized_result() -> None:
    settings = Settings(warmup_enabled=False)
    result = SynthesisResult(
        query="How do I contribute?",
        answer="Fork and open an MR [1].",
        grade_passed=True,
        thread_id="thread-1",
        trace_id="trace-abc",
    )

    with patch("cortex.api.app.KBGraphAgent") as mock_agent_cls:
        mock_agent_cls.return_value.ask.return_value = result
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/v1/ask",
                json={"query": "How do I contribute?"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == result.answer
    assert body["grade_passed"] is True
    assert body["thread_id"] == "thread-1"
    assert body["trace_id"] == "trace-abc"
    mock_agent_cls.return_value.ask.assert_called_once()
