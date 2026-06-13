"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from cortex.api.schemas import AskRequest, HealthResponse, ReadyResponse, WarmupStepResponse
from cortex.api.serialize import synthesis_result_to_dict
from cortex.agent.runner import KBGraphAgent
from cortex.settings import Settings
from cortex.warmup import WarmupReport

router = APIRouter()


def _warmup_response(report: WarmupReport | None) -> ReadyResponse:
    if report is None:
        return ReadyResponse(ready=False, warmup_ms=0, steps=[])
    return ReadyResponse(
        ready=report.ok,
        warmup_ms=report.total_ms,
        steps=[
            WarmupStepResponse(
                name=s.name,
                duration_ms=s.duration_ms,
                ok=s.ok,
                detail=s.detail,
            )
            for s in report.steps
        ],
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    report: WarmupReport | None = getattr(request.app.state, "warmup", None)
    return _warmup_response(report)


@router.post("/ask")
def ask(body: AskRequest, request: Request) -> dict:
    agent: KBGraphAgent | None = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    report: WarmupReport | None = getattr(request.app.state, "warmup", None)
    if report is not None and not report.ok:
        raise HTTPException(status_code=503, detail="Warmup failed; service not ready")

    settings: Settings = request.app.state.settings
    result = agent.ask(
        body.query,
        source_type=body.source_type,
        limit=body.limit,
        department=body.department,
        use_rerank=body.use_rerank,
        use_grader=body.use_grader,
        thread_id=body.thread_id,
    )
    return synthesis_result_to_dict(result, settings)
