"""Request id, access log, body size limit and the last-resort 500 handler (pure ASGI middleware)."""
from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ml.service.errors import error_response
from ml.service.logs import request_id_var

logger = logging.getLogger("ml.service.access")
error_logger = logging.getLogger("ml.service.errors")

REQUEST_ID_HEADER = "X-Request-ID"
# a client-provided id is written to logs as is, so only safe characters are accepted
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def resolve_request_id(given: str | None) -> str:
    if given and _VALID_REQUEST_ID.fullmatch(given.strip()):
        return given.strip()
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(_header(scope, b"x-request-id"))
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status: list[int] = []

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status.append(message["status"])
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            length = _header(scope, b"content-length")
            if length and length.isdigit() and int(length) > self.max_body_bytes:
                response = error_response(413, "payload_too_large",
                                          f"Request body is larger than {self.max_body_bytes} bytes")
                await response(scope, receive, send_with_id)
            else:
                await self.app(scope, receive, send_with_id)
        except Exception:
            # traceback goes to the log only; the client gets a short message and the request id
            error_logger.exception("unhandled error in %s %s", scope.get("method"), scope.get("path"))
            if status:  # the response has already started, nothing can be sent anymore
                raise
            response = error_response(500, "internal_error", "Internal error in the model service")
            await response(scope, receive, send_with_id)
        finally:
            ms = (time.perf_counter() - start) * 1000
            logger.info("%s %s %s %.1fms", scope.get("method"), scope.get("path"),
                        status[0] if status else "-", ms)
            request_id_var.reset(token)
