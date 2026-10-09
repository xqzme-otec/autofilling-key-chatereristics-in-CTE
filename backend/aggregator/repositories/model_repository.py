"""Gateway to the model service (HTTP wrapper around ml.pipeline.Pipeline.analyze)."""
from typing import Protocol

import httpx
from pydantic import ValidationError

from aggregator.domain import (
    AnalysisResult,
    Answer,
    ModelBadResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)

ANALYZE_PATH = "/analyze"


class ModelRepository(Protocol):
    async def analyze(self, text: str, answers: list[Answer], tokens: list[dict] | None) -> AnalysisResult: ...


class HttpModelRepository:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def analyze(self, text: str, answers: list[Answer], tokens: list[dict] | None) -> AnalysisResult:
        payload: dict = {"raw": text, "answers": [a.to_model() for a in answers]}
        if tokens is not None:
            payload["tokens"] = tokens

        try:
            response = await self._client.post(ANALYZE_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError("Model service did not respond in time") from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError(f"Model service is unavailable: {exc}") from exc

        if response.status_code != httpx.codes.OK:
            raise ModelBadResponseError(f"Model service returned HTTP {response.status_code}")
        try:
            return AnalysisResult.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ModelBadResponseError(f"Model service returned an unexpected response: {exc}") from exc
