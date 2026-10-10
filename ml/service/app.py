"""HTTP model service: a thin wrapper around ml.pipeline.Pipeline.analyze() (docs/api.md, section 4).

    POST /analyze   body {"raw", "answers", "tokens"?} -> Pipeline.analyze() result + `known` on new
                    characteristics (ml/service/analysis.py)
    GET  /health    liveness: the process answers; also shows the pipeline and Postgres status
    GET  /ready     readiness: 200 when the pipeline is loaded, 503 while loading or after a failure;
                    Postgres is not required: without it the service degrades, it is not unready

The pipeline is loaded once per process in a background thread at startup, so /health answers
immediately; /analyze returns 503 until it is ready.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml.service.analysis import AnalysisService
from ml.service.config import Settings
from ml.service.errors import SafeJSONResponse, ServiceError, error_response
from ml.service.holder import FAILED, LOADING, READY, Analyzer, PipelineHolder, PipelineNotReady
from ml.service.logs import setup_logging
from ml.service.middleware import RequestContextMiddleware
from ml.service.repository import NewCharacteristicsRepository
from ml.service.schemas import AnalyzeRequest

logger = logging.getLogger(__name__)

_HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}


def default_factory(settings: Settings) -> Callable[[], Analyzer]:
    def build() -> Analyzer:
        from ml.pipeline import Pipeline  # heavy import (torch): only when the real pipeline is needed

        return Pipeline(ktru_path=settings.ktru_path) if settings.ktru_path else Pipeline()

    return build


def create_app(settings: Settings | None = None, pipeline_factory: Callable[[], Analyzer] | None = None,
               load_in_background: bool = True, configure_logging: bool = True,
               repository: NewCharacteristicsRepository | None = None) -> FastAPI:
    """repository: None -> built from settings.postgres_dsn (no DSN: Postgres is off)."""
    settings = settings or Settings.from_env()
    if configure_logging:
        setup_logging(settings.log_level)
    holder = PipelineHolder(pipeline_factory or default_factory(settings))
    if repository is None and settings.postgres_dsn:
        repository = NewCharacteristicsRepository(settings.postgres_dsn)
    service = AnalysisService(holder, repository)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("model service starting: max_text_length=%d max_answers=%d ktru_path=%s postgres=%s",
                    settings.max_text_length, settings.max_answers, settings.ktru_path or "default",
                    "on" if repository is not None else "off")
        if repository is not None:
            repository.open()  # connects in the background: a missing Postgres does not block the start
        if load_in_background:
            holder.load_in_background()
            # the new-characteristic check (rules + classifier) loads right after the pipeline, not on a request
            threading.Thread(target=lambda: holder.wait() and service.warm_up(), name="newchar-loader",
                             daemon=True).start()
        else:
            holder.load()
        yield
        if repository is not None:
            repository.close()
        logger.info("model service stopped")

    app = FastAPI(title="KTRU model service", version="1.0.0", lifespan=lifespan,
                  default_response_class=SafeJSONResponse)
    app.state.settings = settings
    app.state.holder = holder
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError):
        logger.info("request rejected: %s %s: %s", exc.status, exc.code, exc.message)
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        # user input is not echoed back (no "input"/"ctx"): only where and what is wrong
        details = [{"loc": ".".join(str(p) for p in e.get("loc", ())), "msg": e.get("msg", ""),
                    "type": e.get("type", "")} for e in exc.errors()]
        first = details[0] if details else {"loc": "body", "msg": "invalid request"}
        logger.info("invalid request body: %s", "; ".join(f"{d['loc']}: {d['msg']}" for d in details)[:500])
        return error_response(422, "validation_error", f"Invalid request body: {first['loc']}: {first['msg']}",
                              details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, _HTTP_CODES.get(exc.status_code, "http_error"), str(exc.detail),
                              headers=getattr(exc, "headers", None))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # Postgres down is not a failure: /analyze still works, only the known-characteristics step is skipped
        postgres = "off"
        if repository is not None:
            postgres = "ok" if await run_in_threadpool(repository.ping) else "unavailable"
        return {"status": "degraded" if postgres == "unavailable" else "ok", "pipeline": holder.status,
                "postgres": postgres}

    @app.get("/ready")
    async def ready():
        body: dict[str, Any] = {"status": holder.status}
        if holder.status == READY:
            body["load_seconds"] = holder.load_seconds
            return body
        if holder.status == FAILED:
            body["error"] = holder.error
        return SafeJSONResponse(body, status_code=503)

    @app.post("/analyze")
    async def analyze(body: AnalyzeRequest):
        if len(body.raw) > settings.max_text_length:
            raise ServiceError(422, "text_too_long",
                               f"Field 'raw' is longer than {settings.max_text_length} characters "
                               f"({len(body.raw)})")
        if body.answers and len(body.answers) > settings.max_answers:
            raise ServiceError(422, "too_many_answers",
                               f"Field 'answers' has more than {settings.max_answers} items ({len(body.answers)})")
        try:
            # CPU-bound and synchronous: run outside the event loop; the holder serializes pipeline calls
            result = await run_in_threadpool(service.analyze, body.to_pipeline())
        except PipelineNotReady as e:
            message = "Model is still loading, retry later" if e.status == LOADING else "Model failed to load"
            raise ServiceError(503, "model_not_ready", message) from None
        # returned as a response directly: no jsonable_encoder pass, numpy values and NaN are handled
        return SafeJSONResponse(result)

    return app
