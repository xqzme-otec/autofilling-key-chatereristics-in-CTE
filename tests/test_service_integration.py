"""HTTP model service with the real Pipeline (model + data/processed/ktru.parquet).

Slow start (~10 s: model and embeddings). Skip with `pytest -m "not integration"`.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from ml.ktru import DEFAULT_PATH
from ml.service.app import create_app
from ml.service.config import Settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="no data/processed/ktru.parquet"),
]

CODE = re.compile(r"\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}")
TEXTS = [
    "стул ученический, ростовая группа 5",
    "стол письменный",
    "стл на металлическом каркасе, без подлокотников",
    "молоток с деревянным бойком",
    "куртка мужская 48",
    "стул кожзам высота 110 см цвет синий гарантия 3 года",
    "ноутбук игровой",
    "😀 стул",
    "",
]


@pytest.fixture(scope="module")
def client():
    app = create_app(Settings(), load_in_background=False)
    with TestClient(app) as c:
        assert app.state.holder.status == "ready", app.state.holder.error
        yield c


def analyze(client, raw: str, answers: list | None = None, **headers) -> dict:
    r = client.post("/analyze", json={"raw": raw, "answers": answers or []}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def answer_for(next_step: dict, option: str | None = None) -> dict:
    """What the aggregator sends back for a next_step (backend/internal/service/analysis.go, toAnswer)."""
    a = {"type": next_step["type"], "value": option if option is not None else next_step["options"][0]}
    if next_step["type"] == "characteristic":
        a["characteristic"] = next_step["characteristic"]
    return a


# --- aggregator contract ----------------------------------------------------------------------------
# Reimplements what the Go aggregator needs from a /analyze response (origin/aggregator-service:
# backend/internal/client/model/client.go `validate` and backend/internal/domain/domain.go
# `AnalysisResult`): required fields, and JSON types that Go's json.Unmarshal accepts.

def _str(v) -> bool:
    return isinstance(v, str)


def _opt_str(v) -> bool:
    return v is None or isinstance(v, str)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _opt_num(v) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def check_aggregator_contract(r: dict) -> None:
    assert _str(r.get("input")) and _str(r.get("text")), "input and text are required strings"
    for c in r["corrections"]:
        assert _str(c["from"]) and _str(c["to"]) and _opt_str(c.get("reason")), c
    for ch in r["characteristics"]:
        assert _str(ch["name"]) and _str(ch["value"]), ch
        assert _opt_str(ch.get("unit")) and _opt_str(ch.get("source")), ch
    for c in r["candidates"]:
        assert _str(c["code"]) and _str(c["name"]) and _opt_num(c.get("score")), c
    assert _int(r["candidates_total"])
    for e in r["errors"]:
        assert _str(e["type"]) and _str(e["message"]), e
        assert _opt_str(e.get("characteristic")) and _opt_str(e.get("suggestion")), e
    q = r["next_step"]
    if q is not None:
        assert q["type"] in ("product", "characteristic", "code"), q
        assert _str(q.get("message")), "next_step.message is required"
        assert isinstance(q.get("options"), list) and q["options"], "next_step.options is required"
        assert all(_str(o) for o in q["options"]) and _opt_str(q.get("characteristic")), q
        if q["type"] == "characteristic":
            assert _str(q["characteristic"]), "the aggregator copies it into the answer"
    assert _opt_str(r["ktru"]) and _opt_str(r["ktru_name"])
    for s in r["steps"]:
        assert _str(s["stage"]) and _int(s["count"]), s


def test_ready(client):
    ready = client.get("/ready").json()
    assert ready["status"] == "ready"
    assert ready["newchar_classifier"] in ("loaded", "rules_only")
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["pipeline"] == "ready" and health["postgres"] == "off"
    assert health["newchar_classifier"] in ("loaded", "rules_only")


@pytest.mark.parametrize("raw", TEXTS)
def test_response_matches_aggregator_contract(client, raw):
    r = client.post("/analyze", json={"raw": raw, "answers": []})
    assert r.status_code == 200
    body = json.loads(r.text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    check_aggregator_contract(body)
    # full output of ML.md is passed through as is
    for key in ("input", "text", "corrections", "characteristics", "candidates", "candidates_total", "errors",
                "next_step", "ktru", "ktru_name", "steps", "missing_required", "timings_ms"):
        assert key in body


def test_happy_path(client):
    r = analyze(client, "стул ученический, ростовая группа 5")
    assert r["input"] == "стул ученический, ростовая группа 5"
    assert r["candidates_total"] > 0
    assert all(c["name"].startswith("Стул ученический") for c in r["candidates"])
    assert any(ch["name"] == "Ростовая группа" for ch in r["characteristics"]), r["characteristics"]
    assert r["next_step"] is not None or r["ktru"] is not None


def test_dialog_reaches_final_code(client):
    raw, answers = "стол письменный", []
    for _ in range(8):
        r = analyze(client, raw, answers)
        check_aggregator_contract(r)
        if r["ktru"]:
            break
        assert r["next_step"], r["errors"]
        answers.append(answer_for(r["next_step"]))
    assert CODE.fullmatch(r["ktru"]), r
    assert r["ktru_name"] and r["candidates_total"] == 1 and r["next_step"] is None
    assert len(answers) >= 1
    assert any(ch["origin"] == "answer" for ch in r["characteristics"])


def test_missing_required_hint_stays_when_the_code_is_determined(client):
    first = analyze(client, "стол письменный")
    r = analyze(client, "стол письменный", [answer_for(first["next_step"])])
    assert r["ktru"] and r["missing_required"]
    hints = [e for e in r["errors"] if e["type"] == "missing_required"]
    assert len(hints) == 1 and hints[0]["severity"] == "warning"
    assert all(name in hints[0]["message"] for name in r["missing_required"])
    assert first["next_step"]["characteristic"] not in r["missing_required"]  # the answer filled it in


def test_product_answer_narrows_candidates(client):
    first = analyze(client, "куртка мужская 48")
    q = first["next_step"]
    assert q and q["type"] == "product"
    second = analyze(client, "куртка мужская 48", [answer_for(q)])
    assert second["candidates_total"] < first["candidates_total"]
    assert {c["name"] for c in second["candidates"]} == {q["options"][0]}


def test_typo_is_corrected(client):
    r = analyze(client, "стл на металлическом каркасе, без подлокотников")
    assert any(c["from"] == "стл" for c in r["corrections"])


def test_unknown_product(client):
    r = analyze(client, "ноутбук игровой")
    assert r["ktru"] is None and r["next_step"] is None
    assert any(e["type"] == "product_not_found" for e in r["errors"])


def test_empty_text_returns_result_without_code(client):
    r = analyze(client, "   ")
    assert r["ktru"] is None and r["next_step"] is None and r["candidates_total"] == 0


def test_emoji_text(client):
    r = analyze(client, "😀 стул 🪑")
    assert r["input"] == "😀 стул 🪑"
    assert r["candidates_total"] > 0


def test_answer_not_matching_candidates_is_a_warning(client):
    r = analyze(client, "стол письменный", [{"type": "characteristic", "characteristic": "Нет такой", "value": "Да"}])
    assert any(e["type"] == "invalid_answer" and e["severity"] == "warning" for e in r["errors"])


def test_long_text_within_limit(client):
    raw = ", ".join(["стул ученический на металлическом каркасе"] * 100)[:5000]
    assert analyze(client, raw)["input"] == raw


def test_too_long_text_rejected_before_pipeline(client):
    r = client.post("/analyze", json={"raw": "стул " * 2000, "answers": []})
    assert r.status_code == 422 and r.json()["error"]["code"] == "text_too_long"


def test_request_id_in_logs(client, caplog):
    caplog.set_level(logging.INFO)
    analyze(client, "стол письменный", **{"X-Request-ID": "int-trace-1"})
    mine = [r for r in caplog.records if getattr(r, "request_id", None) == "int-trace-1"]
    assert any(r.getMessage().startswith("analyze: ") for r in mine)
    assert any(r.name == "ml.service.access" for r in mine)


def test_concurrent_requests_match_sequential(client):
    """20 parallel requests on one Pipeline: same results as one by one (the holder lock keeps it safe)."""
    def strip(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "timings_ms"}

    texts = [t for t in TEXTS if t] * 3
    expected = {t: strip(analyze(client, t)) for t in set(texts)}
    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda t: strip(analyze(client, t)), texts))
    for t, r in zip(texts, results):
        assert r == expected[t], t
