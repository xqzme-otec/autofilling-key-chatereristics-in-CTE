"""Application entry point and dependency wiring.

    uvicorn aggregator.main:app --port 8000
"""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aggregator.config import Settings
from aggregator.handlers import register_error_handlers, router
from aggregator.repositories import HttpModelRepository, InMemorySessionRepository
from aggregator.services import AnalysisService


def create_app(settings: Settings | None = None, model_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """`model_transport` replaces the network transport to the model service (used in tests)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(base_url=settings.model_service_url,
                                     timeout=settings.model_timeout_s,
                                     transport=model_transport) as client:
            app.state.analysis_service = AnalysisService(
                sessions=InMemorySessionRepository(settings.max_sessions),
                model=HttpModelRepository(client),
                max_text_length=settings.max_text_length,
            )
            yield

    app = FastAPI(title="KTRU autofill aggregator", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    register_error_handlers(app)
    app.include_router(router)

    @app.get("/health", tags=["service"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
