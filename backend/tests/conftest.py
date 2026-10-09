import json

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from aggregator.config import Settings
from aggregator.main import create_app

CODE_1 = "31.01.11.150-00000003"
CODE_2 = "31.01.11.150-00000004"


def model_result(raw: str, answers: list[dict]) -> dict:
    """Response in the format of ml.pipeline.Pipeline.analyze(): one question, then the final code."""
    done = any(a.get("characteristic") == "Наличие подлокотников" for a in answers)
    candidates = [{"code": CODE_1, "name": "Стул на металлическом каркасе", "score": 0.92}]
    if not done:
        candidates.append({"code": CODE_2, "name": "Стул на металлическом каркасе", "score": 0.91})
    return {
        "input": raw,
        "text": raw.replace("стл", "стул"),
        "corrections": [{"from": "стл", "to": "стул", "span": [0, 3],
                         "reason": "опечатка (отличается на 1 симв.)", "alternatives": ["стол"]}],
        "characteristics": [{"name": "Тип каркаса", "value": "Металлический", "unit": None,
                             "source": "металлическом", "span": [8, 21], "method": "stem", "confidence": 0.85}],
        "candidates": candidates,
        "candidates_total": len(candidates),
        "errors": [],
        "next_step": None if done else {
            "type": "characteristic",
            "message": "Подходит 2 кодов КТРУ. Уточните характеристику «Наличие подлокотников»:",
            "characteristic": "Наличие подлокотников",
            "options": ["Да", "Нет"],
            "required": True,
        },
        "ktru": CODE_1 if done else None,
        "ktru_name": "Стул на металлическом каркасе" if done else None,
        "steps": [{"stage": "Поиск по описанию", "count": 2}],
        "timings_ms": {"correct": 0.3, "search": 6.1},
    }


class FakeModel:
    """Stands in for the model service at the HTTP level and records received requests."""

    def __init__(self):
        self.requests: list[dict] = []
        self.handler = self.ok

    def ok(self, payload: dict) -> httpx.Response:
        return httpx.Response(200, json=model_result(payload["raw"], payload["answers"]))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path == "/analyze"
        payload = json.loads(request.content)
        self.requests.append(payload)
        return self.handler(payload)


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def redis_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def client(fake_model, redis_server):
    settings = Settings(model_service_url="http://model", max_text_length=100)
    app = create_app(settings, model_transport=httpx.MockTransport(fake_model),
                     redis=fakeredis.FakeAsyncRedis(server=redis_server, decode_responses=True))
    with TestClient(app) as c:
        yield c
