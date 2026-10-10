"""HTTP model service with a fake pipeline: fast, no model or KTRU data needed."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.service.app import create_app
from ml.service.config import ConfigError, Settings
from ml.service.middleware import resolve_request_id

HEX32 = re.compile(r"[0-9a-f]{32}")


def fake_result(raw: str = "стул") -> dict:
    """Same shape as Pipeline.analyze() (ML.md, «Формат ответа»)."""
    return {
        "input": raw,
        "text": raw,
        "corrections": [],
        "characteristics": [{"name": "Тип каркаса", "value": "Металлический", "unit": None, "source": "металл",
                             "span": [0, 6], "method": "stem", "confidence": 0.85, "origin": "ktru"}],
        "candidates": [{"code": "31.01.11.150-00000003", "name": "Стул", "score": 0.92}],
        "candidates_total": 1,
        "errors": [],
        "next_step": None,
        "ktru": "31.01.11.150-00000003",
        "ktru_name": "Стул",
        "steps": [{"stage": "Поиск по описанию", "count": 23}],
        "missing_required": [],
        "timings_ms": {"correct": 0.1},
    }


class FakePipeline:
    def __init__(self, delay: float = 0.0, error: Exception | None = None, result: dict | None = None):
        self.calls: list[dict] = []
        self.delay, self.error, self.result = delay, error, result
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def analyze(self, request: dict) -> dict:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            logging.getLogger("tests.fake_pipeline").info("fake analyze called")
            self.calls.append(request)
            if self.delay:
                time.sleep(self.delay)
            if self.error:
                raise self.error
            return self.result if self.result is not None else fake_result(request["raw"])
        finally:
            with self._guard:
                self.active -= 1


def client_for(pipeline=None, settings: Settings | None = None, factory=None, background: bool = False) -> TestClient:
    pipeline = pipeline or FakePipeline()
    app = create_app(settings or Settings(), pipeline_factory=factory or (lambda: pipeline),
                     load_in_background=background)
    return TestClient(app)


@pytest.fixture
def fake():
    return FakePipeline()


@pytest.fixture
def client(fake):
    with client_for(fake) as c:
        yield c


# --- health / readiness ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "pipeline": "ready"}


def test_ready_when_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert isinstance(body["load_seconds"], float)


def test_while_loading_health_answers_and_analyze_is_503():
    release = threading.Event()
    pipeline = FakePipeline()

    def slow_factory():
        release.wait(5)
        return pipeline

    with client_for(factory=slow_factory, background=True) as c:
        assert c.get("/health").json() == {"status": "ok", "pipeline": "loading"}
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        r = c.post("/analyze", json={"raw": "стул", "answers": []})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_not_ready"
        assert "loading" in r.json()["error"]["message"]

        release.set()
        assert c.app.state.holder.wait(5)
        assert c.get("/ready").status_code == 200
        assert c.post("/analyze", json={"raw": "стул", "answers": []}).status_code == 200


def test_failed_load_is_reported(caplog):
    def broken_factory():
        raise FileNotFoundError("ktru.parquet not found")

    caplog.set_level(logging.INFO)
    with client_for(factory=broken_factory) as c:
        assert c.get("/health").json() == {"status": "ok", "pipeline": "failed"}
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "failed"
        assert "FileNotFoundError" in r.json()["error"]
        r = c.post("/analyze", json={"raw": "стул", "answers": []})
        assert r.status_code == 503
        assert r.json()["error"] == {"code": "model_not_ready", "message": "Model failed to load",
                                     "request_id": r.headers["x-request-id"]}
    assert any(rec.exc_info and "failed to load" in rec.getMessage() for rec in caplog.records)


# --- happy path and payload mapping ---------------------------------------------------------------

def test_analyze_returns_pipeline_result_as_is(client, fake):
    r = client.post("/analyze", json={"raw": "стул", "answers": []})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == fake_result("стул")
    assert fake.calls == [{"raw": "стул", "answers": []}]


def test_aggregator_payload_is_passed_to_pipeline(client, fake):
    payload = {"raw": "стл на металлическом каркасе", "answers": [
        {"type": "product", "value": "Стул на металлическом каркасе"},
        {"type": "characteristic", "characteristic": "Наличие подлокотников", "value": "Нет"},
        {"type": "code", "value": "31.01.11.150-00000003"},
    ]}
    assert client.post("/analyze", json=payload).status_code == 200
    assert fake.calls[-1] == payload


def test_missing_or_null_answers_mean_no_answers(client, fake):
    assert client.post("/analyze", json={"raw": "стул"}).status_code == 200
    assert client.post("/analyze", json={"raw": "стул", "answers": None}).status_code == 200
    assert [c["answers"] for c in fake.calls] == [[], []]


def test_answer_type_defaults_to_characteristic(client, fake):
    r = client.post("/analyze", json={"raw": "стул", "answers": [{"characteristic": "Цвет", "value": "Синий"}]})
    assert r.status_code == 200
    assert fake.calls[-1]["answers"] == [{"type": "characteristic", "characteristic": "Цвет", "value": "Синий"}]


def test_tokens_and_unknown_fields(client, fake):
    tokens = [{"text": "стул", "lemma": "стул", "span": [0, 4], "known": True}]
    r = client.post("/analyze", json={"raw": "стул", "answers": [], "tokens": tokens, "session": "x"})
    assert r.status_code == 200
    assert fake.calls[-1] == {"raw": "стул", "answers": [], "tokens": tokens}


def test_empty_and_whitespace_raw_reach_the_pipeline(client, fake):
    for raw in ("", "   "):
        assert client.post("/analyze", json={"raw": raw, "answers": []}).status_code == 200
    assert [c["raw"] for c in fake.calls] == ["", "   "]


def test_unicode_and_emoji_round_trip(client, fake):
    raw = "стул 😀 «ёлка» — 漢字   х/б"
    r = client.post("/analyze", json={"raw": raw, "answers": []})
    assert r.status_code == 200
    assert fake.calls[-1]["raw"] == raw
    assert r.json()["input"] == raw
    assert "😀".encode() in r.content  # UTF-8, not \u-escaped


# --- validation -------------------------------------------------------------------------------------

@pytest.mark.parametrize("payload, loc", [
    ({}, "body.raw"),
    ({"answers": []}, "body.raw"),
    ({"raw": None}, "body.raw"),
    ({"raw": 123}, "body.raw"),
    ({"raw": ["стул"]}, "body.raw"),
    ({"raw": "стул", "answers": "Да"}, "body.answers"),
    ({"raw": "стул", "answers": {"type": "code"}}, "body.answers"),
    ({"raw": "стул", "answers": ["Да"]}, "body.answers.0"),
    ({"raw": "стул", "answers": [{"type": "characteristic", "value": "Да"}]}, "body.answers.0"),
    ({"raw": "стул", "answers": [{"type": "characteristic", "characteristic": "", "value": "Да"}]}, "body.answers.0"),
    ({"raw": "стул", "answers": [{"type": "product"}]}, "body.answers.0.value"),
    ({"raw": "стул", "answers": [{"type": "product", "value": 5}]}, "body.answers.0.value"),
    ({"raw": "стул", "answers": [{"type": "unknown", "value": "x"}]}, "body.answers.0.type"),
    ({"raw": "стул", "tokens": "стул"}, "body.tokens"),
])
def test_invalid_body_is_422(client, fake, payload, loc):
    r = client.post("/analyze", json=payload, headers={"X-Request-ID": "req-422"})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == "req-422"
    assert error["message"].startswith("Invalid request body: ")
    assert any(d["loc"].startswith(loc) for d in error["details"]), error["details"]
    assert all(set(d) == {"loc", "msg", "type"} for d in error["details"])
    assert fake.calls == []


def test_validation_error_does_not_echo_user_text(client):
    secret = "секретное описание товара 42"
    r = client.post("/analyze", json={"raw": 1, "answers": [{"type": "product", "value": 1, "x": secret}]})
    assert r.status_code == 422
    assert secret not in r.text


@pytest.mark.parametrize("body", [b"{not json", b"", b"[1, 2]", b'"\xd1\x81\xd1\x82\xd1\x83\xd0\xbb"', b"null"])
def test_garbage_json_is_422(client, fake, body):
    r = client.post("/analyze", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "validation_error"
    assert fake.calls == []


def test_form_body_is_422(client):
    r = client.post("/analyze", data={"raw": "стул"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_text_too_long():
    fake = FakePipeline()
    with client_for(fake, Settings(max_text_length=10)) as c:
        assert c.post("/analyze", json={"raw": "я" * 10, "answers": []}).status_code == 200
        r = c.post("/analyze", json={"raw": "я" * 11, "answers": []})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "text_too_long"
        assert "10 characters" in r.json()["error"]["message"]
    assert len(fake.calls) == 1


def test_default_max_text_length_is_above_aggregator_limit(client):
    assert client.post("/analyze", json={"raw": "с" * 2000, "answers": []}).status_code == 200
    r = client.post("/analyze", json={"raw": "с" * 5001, "answers": []})
    assert r.status_code == 422 and r.json()["error"]["code"] == "text_too_long"


def test_too_many_answers():
    with client_for(settings=Settings(max_answers=2)) as c:
        answers = [{"type": "code", "value": "1"}] * 3
        r = c.post("/analyze", json={"raw": "стул", "answers": answers})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "too_many_answers"


def test_body_too_large_is_413():
    fake = FakePipeline()
    with client_for(fake, Settings(max_body_bytes=100)) as c:
        r = c.post("/analyze", json={"raw": "я" * 200, "answers": []}, headers={"X-Request-ID": "big"})
        assert r.status_code == 413
        assert r.json()["error"] == {"code": "payload_too_large", "message": "Request body is larger than 100 bytes",
                                     "request_id": "big"}
        assert r.headers["x-request-id"] == "big"
    assert fake.calls == []


def test_unknown_route_and_method(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
    r = client.get("/analyze")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "method_not_allowed"


# --- request id -------------------------------------------------------------------------------------

def test_request_id_is_echoed(client):
    r = client.post("/analyze", json={"raw": "стул"}, headers={"X-Request-ID": "abc-123.x:y_z"})
    assert r.headers["x-request-id"] == "abc-123.x:y_z"
    assert client.get("/health", headers={"x-request-id": "h1"}).headers["x-request-id"] == "h1"


def test_request_id_is_generated_when_absent(client):
    ids = {client.get("/health").headers["x-request-id"] for _ in range(3)}
    assert len(ids) == 3
    assert all(HEX32.fullmatch(i) for i in ids)


@pytest.mark.parametrize("bad", ["", "   ", "has space", "x" * 129, "line\\nbreak", "кириллица"])
def test_unsafe_request_id_is_replaced(bad):
    assert HEX32.fullmatch(resolve_request_id(bad))


def test_unsafe_request_id_header_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "a b <script>"})
    assert HEX32.fullmatch(r.headers["x-request-id"])


def test_request_id_in_logs_including_pipeline_thread(client, caplog):
    caplog.set_level(logging.INFO)
    client.post("/analyze", json={"raw": "стул"}, headers={"X-Request-ID": "trace-1"})
    pipeline_logs = [r for r in caplog.records if r.name == "tests.fake_pipeline"]
    assert pipeline_logs and all(r.request_id == "trace-1" for r in pipeline_logs)
    access = [r for r in caplog.records if r.name == "ml.service.access"]
    assert access[-1].request_id == "trace-1"
    assert re.fullmatch(r"POST /analyze 200 \d+\.\dms", access[-1].getMessage())
    summary = [r for r in caplog.records if r.getMessage().startswith("analyze: ")]
    assert summary and summary[-1].request_id == "trace-1"
    for key in ("raw_len=4", "candidates_total=1", "ktru=31.01.11.150-00000003", "characteristics=1", "errors=0",
                "search_candidates=23"):
        assert key in summary[-1].getMessage()
    # outside of a request the id is "-"
    logging.getLogger("tests").info("outside")
    assert caplog.records[-1].request_id == "-"


def test_full_text_is_not_logged_at_info(client, caplog):
    caplog.set_level(logging.INFO)
    raw = "уникальное описание товара для проверки логов " * 5
    client.post("/analyze", json={"raw": raw})
    assert all(raw.strip() not in r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)


def test_full_text_is_logged_at_debug(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post("/analyze", json={"raw": "отладочный текст"})
    assert any("отладочный текст" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


# --- errors -----------------------------------------------------------------------------------------

def test_unhandled_error_is_500_without_traceback(caplog):
    caplog.set_level(logging.INFO)
    fake = FakePipeline(error=RuntimeError("secret internal detail"))
    with client_for(fake) as c:
        r = c.post("/analyze", json={"raw": "стул"}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Internal error in the model service",
                                  "request_id": "boom-1"}}
    assert "secret" not in r.text and "Traceback" not in r.text
    assert r.headers["x-request-id"] == "boom-1"
    logged = [rec for rec in caplog.records if rec.exc_info]
    assert logged and logged[-1].request_id == "boom-1"
    assert "secret internal detail" in str(logged[-1].exc_info[1])
    access = [rec for rec in caplog.records if rec.name == "ml.service.access"]
    assert access[-1].getMessage().startswith("POST /analyze 500 ")


def test_service_keeps_working_after_an_error():
    fake = FakePipeline(error=ValueError("once"))
    with client_for(fake) as c:
        assert c.post("/analyze", json={"raw": "стул"}).status_code == 500
        fake.error = None
        assert c.post("/analyze", json={"raw": "стул"}).status_code == 200


def test_numpy_and_nan_values_are_serialized():
    result = fake_result()
    result["candidates"] = [{"code": "1", "name": "Стул", "score": np.float32(0.5)},
                            {"code": "2", "name": "Стол", "score": float("nan")}]
    result["candidates_total"] = np.int64(2)
    result["steps"] = [{"stage": "x", "count": np.int32(3)}]
    with client_for(FakePipeline(result=result)) as c:
        r = c.post("/analyze", json={"raw": "стул"})
    assert r.status_code == 200
    body = json.loads(r.text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert body["candidates"][0]["score"] == 0.5
    assert body["candidates"][1]["score"] is None
    assert body["candidates_total"] == 2 and body["steps"][0]["count"] == 3


# --- concurrency ------------------------------------------------------------------------------------

def test_parallel_requests_are_serialized():
    fake = FakePipeline(delay=0.02)
    with client_for(fake) as c:
        def call(i: int):
            return c.post("/analyze", json={"raw": f"стул {i}"}, headers={"X-Request-ID": f"p{i}"})

        with ThreadPoolExecutor(max_workers=20) as pool:
            responses = list(pool.map(call, range(20)))
        # liveness is not blocked by the queue
        assert c.get("/health").status_code == 200
    assert [r.status_code for r in responses] == [200] * 20
    assert [r.json()["input"] for r in responses] == [f"стул {i}" for i in range(20)]
    assert [r.headers["x-request-id"] for r in responses] == [f"p{i}" for i in range(20)]
    assert fake.max_active == 1  # Pipeline.analyze is not thread-safe


# --- configuration ----------------------------------------------------------------------------------

def test_settings_defaults():
    s = Settings.from_env({})
    assert (s.host, s.port, s.log_level, s.ktru_path, s.max_text_length) == ("0.0.0.0", 8001, "INFO", None, 5000)


def test_settings_from_env():
    s = Settings.from_env({"LOG_LEVEL": "debug", "PORT": "9000", "KTRU_PATH": "/x/ktru.parquet",
                           "MAX_TEXT_LENGTH": "100", "HOST": "127.0.0.1"})
    assert (s.log_level, s.port, str(s.ktru_path), s.max_text_length, s.host) == \
           ("DEBUG", 9000, "/x/ktru.parquet", 100, "127.0.0.1")


@pytest.mark.parametrize("env", [{"LOG_LEVEL": "LOUD"}, {"PORT": "abc"}, {"MAX_TEXT_LENGTH": "0"},
                                 {"MAX_TEXT_LENGTH": "-5"}, {"MAX_BODY_BYTES": "1.5"}])
def test_settings_invalid(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)
