"""Shared dependencies injected into LangGraph node config."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from cortex.grading.grader import ContextGrader
from cortex.llm.ollama import OllamaChatClient
from cortex.retrieval.pipeline import RetrievalPipeline
from cortex.settings import Settings


@dataclass
class AgentDeps:
    settings: Settings
    pipeline: RetrievalPipeline
    grader: ContextGrader
    llm: OllamaChatClient

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AgentDeps:
        s = settings or Settings()
        return cls(
            settings=s,
            pipeline=RetrievalPipeline(s),
            grader=ContextGrader(s),
            llm=OllamaChatClient(s),
        )


def get_deps(config: RunnableConfig | None) -> AgentDeps:
    configurable = (config or {}).get("configurable") or {}
    deps = configurable.get("deps")
    if deps is None:
        raise RuntimeError("AgentDeps missing from graph config.configurable.deps")
    return deps
