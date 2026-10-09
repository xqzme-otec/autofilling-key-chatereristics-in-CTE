import httpx
import pytest

from tests.conftest import CODE_1

TEXT = "стл на металлическом каркасе"
MODEL_ANSWER = {"type": "characteristic", "characteristic": "Наличие подлокотников", "value": "Нет"}


def start(client) -> dict:
    response = client.post("/api/v1/analyze", json={"text": TEXT})
    assert response.status_code == 200, response.text
    return response.json()


def answer(client, session_id: str, value: str) -> httpx.Response:
    return client.post(f"/api/v1/sessions/{session_id}/answers", json={"value": value})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_returns_agreed_form(client, fake_model):
    body = start(client)

    assert set(body) == {"session_id", "corrections", "characteristics", "candidates", "errors",
                         "next_step", "ktru", "ktru_name", "steps"}
    assert body["corrections"] == [{"from": "стл", "to": "стул", "reason": "опечатка (отличается на 1 симв.)"}]
    assert body["characteristics"] == [{"name": "Тип каркаса", "value": "Металлический", "source": "металлическом"}]
    assert body["candidates"][0] == {"code": CODE_1, "name": "Стул на металлическом каркасе", "score": 0.92}
    assert body["next_step"] == {
        "message": "Подходит 2 кодов КТРУ. Уточните характеристику «Наличие подлокотников»:",
        "characteristic": "Наличие подлокотников",
        "options": ["Да", "Нет"],
    }
    assert body["ktru"] is None
    assert body["steps"] == [{"stage": "Поиск по описанию", "count": 2}]
    assert fake_model.requests == [{"raw": TEXT, "answers": []}]


def test_characteristic_unit_and_error_fields(client, fake_model):
    def with_unit_and_error(payload):
        result = fake_model.ok(payload).json()
        result["characteristics"].append({"name": "Высота", "value": "1000–1200", "unit": "Миллиметр",
                                          "user_value": "1100 Миллиметр", "source": "110 см", "span": [0, 6],
                                          "method": "number", "confidence": 0.95})
        result["errors"] = [{"type": "nonexistent_value", "characteristic": "Материал", "value": "дерево",
                             "source": "дерево", "span": [0, 6], "message": "Значение недопустимо",
                             "suggestion": "Металл"}]
        return httpx.Response(200, json=result)

    fake_model.handler = with_unit_and_error
    body = start(client)

    assert body["characteristics"][1] == {"name": "Высота", "value": "1000–1200", "unit": "Миллиметр",
                                          "source": "110 см"}
    assert body["errors"] == [{"type": "nonexistent_value", "message": "Значение недопустимо",
                               "characteristic": "Материал", "suggestion": "Металл"}]


def test_answer_resends_text_with_all_answers(client, fake_model):
    session_id = start(client)["session_id"]

    response = answer(client, session_id, "Нет")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == session_id
    assert body["next_step"] is None
    assert body["ktru"] == CODE_1
    assert body["ktru_name"] == "Стул на металлическом каркасе"
    assert fake_model.requests[-1] == {"raw": TEXT, "answers": [MODEL_ANSWER]}


def test_product_answer_has_no_characteristic(client, fake_model):
    def product_question(payload):
        result = fake_model.ok(payload).json()
        result["next_step"] = {"type": "product", "message": "Уточните, какой товар нужен:",
                               "characteristic": None, "options": ["Стул", "Табурет"], "counts": [3, 1]}
        return httpx.Response(200, json=result)

    fake_model.handler = product_question
    session_id = start(client)["session_id"]

    assert answer(client, session_id, "Табурет").status_code == 200
    assert fake_model.requests[-1]["answers"] == [{"type": "product", "value": "Табурет"}]


def test_answer_must_be_an_offered_option(client):
    session_id = start(client)["session_id"]

    response = answer(client, session_id, "Может быть")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_answer"


@pytest.mark.parametrize("payload", [{}, {"value": ""}])
def test_answer_validation(client, payload):
    session_id = start(client)["session_id"]

    response = client.post(f"/api/v1/sessions/{session_id}/answers", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_answer_after_done_is_conflict(client):
    session_id = start(client)["session_id"]
    answer(client, session_id, "Нет")

    response = answer(client, session_id, "Нет")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_pending_question"


def test_unknown_route_uses_error_format(client):
    response = client.get("/api/v1/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_unknown_session(client):
    response = answer(client, "nope", "Нет")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


@pytest.mark.parametrize("payload, code", [
    ({}, "validation_error"),
    ({"text": ""}, "validation_error"),
    ({"text": "   "}, "invalid_text"),
    ({"text": "x" * 101}, "invalid_text"),
])
def test_invalid_text(client, fake_model, payload, code):
    response = client.post("/api/v1/analyze", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert fake_model.requests == []


def _raise(exc):
    def handler(_):
        raise exc
    return handler


@pytest.mark.parametrize("handler, status, code", [
    (_raise(httpx.ConnectError("refused")), 503, "model_unavailable"),
    (_raise(httpx.ReadTimeout("slow")), 504, "model_timeout"),
    (lambda _: httpx.Response(500, text="boom"), 502, "model_bad_response"),
    (lambda _: httpx.Response(200, text="not json"), 502, "model_bad_response"),
    (lambda _: httpx.Response(200, json={"unexpected": True}), 502, "model_bad_response"),
])
def test_model_failures(client, fake_model, handler, status, code):
    fake_model.handler = handler

    response = client.post("/api/v1/analyze", json={"text": TEXT})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
