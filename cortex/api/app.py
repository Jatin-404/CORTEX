"""FastAPI application with blocking startup warmup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from cortex.agent.runner import KBGraphAgent
from cortex.api.routes import router
from cortex.logging_config import configure_logging
from cortex.settings import Settings, settings
from cortex.warmup import run_warmup

log = logging.getLogger(__name__)


def create_app(app_settings: Settings | None = None) -> FastAPI:
    app_settings = app_settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = app_settings
        app.state.agent = None
        app.state.warmup = None

        if app_settings.warmup_enabled:
            log.info("warmup_starting")
            report = run_warmup(app_settings)
            app.state.warmup = report
            if not report.ok:
                failed = [s.name for s in report.steps if not s.ok]
                raise RuntimeError(f"Warmup failed at steps: {', '.join(failed)}")
            log.info("warmup_done", extra={"total_ms": report.total_ms})
        else:
            log.info("warmup_skipped")

        app.state.agent = KBGraphAgent(app_settings)
        yield

    app = FastAPI(
        title="Cortex",
        description="Agentic RAG platform — handbook Q&A API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/v1")
    return app


app = create_app(settings)


def main() -> None:
    configure_logging(settings.log_level)
    uvicorn.run(
        "cortex.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
