"""KB answer synthesis — LangGraph orchestration with Corrective-RAG."""

from __future__ import annotations

import logging

from cortex.models.enums import SourceType
from cortex.settings import Settings
from cortex.synthesis.result import SynthesisResult

log = logging.getLogger(__name__)


class KBSynthesizer:
    """
    Public API for KB Q&A — delegates to the LangGraph agent.

    Graph: retrieve -> rerank -> grade -> synthesize (with rewrite loop).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        from cortex.agent.runner import KBGraphAgent

        self.settings = settings or Settings()
        self._agent = KBGraphAgent(self.settings)

    def ask(
        self,
        query: str,
        *,
        source_type: SourceType = SourceType.HANDBOOK_MARKDOWN,
        limit: int | None = None,
        department: str | None = None,
        use_rerank: bool | None = None,
        use_grader: bool | None = None,
        thread_id: str | None = None,
    ) -> SynthesisResult:
        return self._agent.ask(
            query,
            source_type=source_type,
            limit=limit,
            department=department,
            use_rerank=use_rerank,
            use_grader=use_grader,
            thread_id=thread_id,
        )
