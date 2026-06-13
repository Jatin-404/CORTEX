"""Ollama chat client for synthesis and future agent nodes."""

from __future__ import annotations

import logging

import httpx

from cortex.settings import Settings

log = logging.getLogger(__name__)


class OllamaChatClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    temperature
                    if temperature is not None
                    else self.settings.llm_temperature
                ),
            },
        }

        with httpx.Client(
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.llm_timeout_seconds,
        ) as client:
            response = client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        content = data.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Ollama returned empty response")
        return content.strip()
