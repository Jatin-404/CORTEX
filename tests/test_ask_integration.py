"""Integration test for cortex-ask — requires Qdrant, Ollama embed + chat models."""

from __future__ import annotations

import httpx
import pytest

from cortex.settings import Settings
from cortex.synthesis.synthesizer import KBSynthesizer


def _services_available() -> bool:
    settings = Settings()
    try:
        with httpx.Client(timeout=5.0) as client:
            if not client.get(f"{settings.qdrant_url}/collections").is_success:
                return False
            tags = client.get(f"{settings.ollama_base_url}/api/tags").json().get("models", [])
            names = {m.get("name", "") for m in tags}
            has_embed = any("nomic-embed" in n for n in names)
            has_llm = any(settings.llm_model.split(":")[0] in n for n in names) or settings.llm_model in names
            return has_embed and has_llm
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _services_available(),
    reason="Qdrant/Ollama or required models not reachable",
)


def test_ask_contributing_question() -> None:
    synthesizer = KBSynthesizer(Settings())
    result = synthesizer.ask(
        "How do I contribute to the GitLab handbook?",
        limit=5,
        use_rerank=True,
    )
    assert result.answer
    assert len(result.answer) > 50
    assert result.sources
    paths = [s.relative_path for s in result.sources]
    assert any("contributing" in p for p in paths)
    assert "[" in result.answer  # inline citation expected
