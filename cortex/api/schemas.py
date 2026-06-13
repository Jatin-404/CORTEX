"""Pydantic models for the Cortex REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortex.models.enums import SourceType


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    source_type: SourceType = SourceType.HANDBOOK_MARKDOWN
    limit: int | None = Field(default=None, ge=1, le=50)
    department: str | None = None
    use_rerank: bool | None = None
    use_grader: bool | None = None
    thread_id: str | None = None


class WarmupStepResponse(BaseModel):
    name: str
    duration_ms: int
    ok: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    ready: bool
    warmup_ms: int
    steps: list[WarmupStepResponse]


class HealthResponse(BaseModel):
    status: str
