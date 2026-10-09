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


class ModelRepository(Protocol):
    async def analyze(self, text: str, answers: list[Answer]) -> AnalysisResult: ...


class HttpModelRepository:
    def __init__(self, client: httpx.AsyncClient, analyze_path: str):
        self._client = client
        self._analyze_path = analyze_path

    async def analyze(self, text: str, answers: list[Answer]) -> AnalysisResult:
        payload = {"raw": text, "answers": [a.to_model() for a in answers]}

        try:
            response = await self._client.post(self._analyze_path, json=payload)
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
