"""Error responses: {"error": {"code", "message", "request_id", "details"?}}.

Same shape as the aggregator's errors (docs/api.md, «Ошибки»), plus request_id to find the log lines.
"""
from __future__ import annotations

import json
import math
from typing import Any

from starlette.responses import JSONResponse

from ml.service.logs import request_id_var


def _default(o: Any) -> Any:
    """numpy scalars/arrays (and other number-like objects) -> plain JSON values."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _finite(o: Any) -> Any:
    """NaN/Infinity are not valid JSON (the Go aggregator would reject the whole body) -> null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        try:
            body = json.dumps(content, ensure_ascii=False, allow_nan=False, default=_default)
        except ValueError:  # NaN or Infinity somewhere in the result
            body = json.dumps(_finite(json.loads(json.dumps(content, default=_default))),
                              ensure_ascii=False, allow_nan=False)
        return body.encode("utf-8")


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_response(status: int, code: str, message: str, details: list | None = None,
                   headers: dict[str, str] | None = None) -> SafeJSONResponse:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details:
        error["details"] = details
    return SafeJSONResponse({"error": error}, status_code=status, headers=headers)
