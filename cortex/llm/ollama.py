"""Ollama chat client for synthesis and future agent nodes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from langfuse import observe

from cortex.observability.langfuse import configure, is_enabled, update_current_generation
from cortex.settings import Settings

if TYPE_CHECKING:
    pass

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
        trace_name: str = "ollama-chat",
    ) -> str:
        if is_enabled(self.settings):
            configure(self.settings)
            return self._chat_traced(
                messages,
                model=model,
                temperature=temperature,
                trace_name=trace_name,
            )
        return self._chat_raw(messages, model=model, temperature=temperature)

    @observe(as_type="generation", capture_input=False, capture_output=False)
    def _chat_traced(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        trace_name: str = "ollama-chat",
    ) -> str:
        model_name = model or self.settings.llm_model
        user_content = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        update_current_generation(
            model=model_name,
            input={
                "user_message": user_content[:500],
                "message_count": len(messages),
            },
            metadata={"temperature": temperature, "trace_name": trace_name},
        )
        content, token_meta = self._chat_raw(
            messages,
            model=model_name,
            temperature=temperature,
        )
        update_current_generation(output=content, metadata=token_meta)
        return content

    def _chat_raw(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str | tuple[str, dict]:
        model_name = model or self.settings.llm_model
        temp = (
            temperature
            if temperature is not None
            else self.settings.llm_temperature
        )

        payload: dict = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp},
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

        content = content.strip()
        token_meta = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
        if is_enabled(self.settings):
            return content, token_meta
        return content
