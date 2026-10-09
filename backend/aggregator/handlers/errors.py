"""Maps domain errors to HTTP responses with a single error body format."""
import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aggregator.domain import (
    DomainError,
    InvalidAnswerError,
    InvalidTextError,
    ModelBadResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
    NoPendingQuestionError,
    SessionNotFoundError,
    StorageUnavailableError,
)

log = logging.getLogger(__name__)

STATUS_BY_ERROR: dict[type[DomainError], int] = {
    InvalidTextError: 422,
    InvalidAnswerError: 422,
    SessionNotFoundError: 404,
    NoPendingQuestionError: 409,
    ModelBadResponseError: 502,
    ModelUnavailableError: 503,
    ModelTimeoutError: 504,
    StorageUnavailableError: 503,
}


def _error(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    body = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status, content={"error": body})


async def _domain_error(_: Request, exc: DomainError) -> JSONResponse:
    status = STATUS_BY_ERROR.get(type(exc), 500)
    if status >= 500:
        log.error("%s: %s", exc.code, exc.message)
    return _error(status, exc.code, exc.message)


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "validation_error", "Request validation failed", jsonable_encoder(exc.errors()))


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    return _error(exc.status_code, code, str(exc.detail))


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _domain_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
