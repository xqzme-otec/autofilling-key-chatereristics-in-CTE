"""Application entry point and dependency wiring.

    uvicorn aggregator.main:app --port 8000
"""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from aggregator.config import Settings
from aggregator.handlers import analysis, register_error_handlers
from aggregator.handlers.schemas import ErrorResponse, SessionResponse
from aggregator.repositories import HttpModelRepository, RedisSessionRepository
from aggregator.services import AnalysisService

API_PREFIX = "/api/v1"
MODEL_ANALYZE_PATH = "/analyze"
ERRORS = {code: {"model": ErrorResponse} for code in (404, 409, 422, 502, 503, 504)}


def create_app(
    settings: Settings | None = None,
    model_transport: httpx.AsyncBaseTransport | None = None,
    redis: Redis | None = None,
) -> FastAPI:
    """`model_transport` and `redis` replace the real model service and Redis (used in tests)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis_client = redis or Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()  # fail fast if Redis is unreachable
        async with httpx.AsyncClient(base_url=settings.model_service_url,
                                     timeout=settings.model_timeout_s,
                                     transport=model_transport) as client:
            app.state.analysis_service = AnalysisService(
                sessions=RedisSessionRepository(redis_client, settings.session_ttl_s),
                model=HttpModelRepository(client, MODEL_ANALYZE_PATH),
                max_text_length=settings.max_text_length,
            )
            try:
                yield
            finally:
                await redis_client.aclose()

    app = FastAPI(title="KTRU autofill aggregator", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    register_error_handlers(app)

    app.add_api_route(f"{API_PREFIX}/analyze", analysis.analyze, methods=["POST"],
                      response_model=SessionResponse, responses=ERRORS, tags=["analysis"])
    app.add_api_route(f"{API_PREFIX}/sessions/{{session_id}}/answers", analysis.answer, methods=["POST"],
                      response_model=SessionResponse, responses=ERRORS, tags=["analysis"])

    @app.get("/health", tags=["service"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
