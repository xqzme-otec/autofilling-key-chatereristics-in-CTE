"""Known new characteristics: ml/service/analysis.py (service layer) and ml/service/repository.py (Postgres).

Without Postgres: the key, the fit check (in-memory repository, ml/fit.py), the degraded mode (unreachable
Postgres) and health. With Postgres: set TEST_POSTGRES_DSN, e.g.
    docker run -d --name ktru-test-pg -p 5432:5432 -e POSTGRES_PASSWORD=test postgres:18
    TEST_POSTGRES_DSN=postgresql://postgres:test@localhost:5432/postgres pytest tests/test_new_characteristics.py
"""
from __future__ import annotations

import copy
import logging
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from ml.service.analysis import characteristic_key
from ml.service.app import create_app
from ml.service.config import Settings
from ml.service.repository import NewCharacteristicsRepository

DSN = os.environ.get("TEST_POSTGRES_DSN")
UNREACHABLE_DSN = "postgresql://nobody:nothing@127.0.0.1:1/none"


def result_with(category: str, *new_names: str) -> dict:
    """Pipeline.analyze() shape: one KTRU characteristic and the given origin=new ones."""
    return {
        "input": "", "text": "", "corrections": [], "product_intent": "молоток", "category": category,
        "characteristics": [{"name": "Масса молотка", "value": "0.6–0.8", "unit": "Килограмм", "origin": "ktru",
                             "confidence": 0.95}]
                           + [{"name": n, "value": "Да", "unit": None, "origin": "new", "confidence": 0.5}
                              for n in new_names],
        "candidates": [{"code": "25.73.30.141-00000001", "name": "Молоток", "score": 0.9}], "candidates_total": 1,
        "errors": [], "next_step": None, "ktru": None, "ktru_name": None,
        "steps": [{"stage": "Поиск по описанию", "count": 5}], "missing_required": [], "timings_ms": {},
    }


class FixedPipeline:
    def __init__(self, result: dict, store=None):
        self.result = result
        if store is not None:  # the fit check reads the KTRU store of the loaded pipeline
            self.store = store

    def analyze(self, request: dict) -> dict:
        # a fresh copy per call (the service marks characteristics in place); the text identifies the description
        return {**copy.deepcopy(self.result), "input": request["raw"], "text": request["raw"]}


class MemoryRepository:
    """The NewCharacteristicsRepository contract in memory: rows (category, product, key, text_hash)."""

    def __init__(self):
        self.rows: set[tuple[str, str, str, str]] = set()

    def open(self) -> None: ...

    def close(self) -> None: ...

    def ping(self) -> bool:
        return True

    def seen_before(self, category: str, product: str, keys: list[str], text_hash: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c, p, key, h in self.rows:
            if (c, p) == (category, product) and key in keys and h != text_hash:
                counts[key] = counts.get(key, 0) + 1
        return counts

    def save(self, category: str, product: str, text_hash: str, items: list[tuple]) -> None:
        self.rows |= {(category, product, key, text_hash) for key, *_ in items}


class Store:
    """KtruStore shape for the fit check: one hammer position without electrical characteristics."""
    items = [{"code": "25.73.30.141-00000001", "name": "Молоток", "category": "инструменты",
              "characteristics": [{"name": "Масса молотка", "unit": "Килограмм", "values": []}]}]


def client(result: dict, repository, store=None) -> TestClient:
    return TestClient(create_app(Settings(), pipeline_factory=lambda: FixedPipeline(result, store),
                                 load_in_background=False, repository=repository))


def new_characteristics(c: TestClient, raw: str) -> list[dict]:
    r = c.post("/analyze", json={"raw": raw, "answers": []})
    assert r.status_code == 200, r.text
    return [ch for ch in r.json()["characteristics"] if ch["origin"] == "new"]


def test_key_ignores_word_forms_order_and_case():
    assert characteristic_key("Противоскользящая рукоятка") == "противоскользящий рукоятка"
    assert characteristic_key("рукоятки противоскользящей") == "противоскользящий рукоятка"


def test_without_a_ktru_store_every_characteristic_fits():
    repository = MemoryRepository()
    with client(result_with("инструменты", "Противоскользящая рукоятка", "Магнитный держатель"), repository) as c:
        assert [ch["known"] for ch in new_characteristics(c, "первое описание")] == [False, False]
        assert [ch["known"] for ch in new_characteristics(c, "второе описание")] == [True, True]
    assert len(repository.rows) == 4  # 2 characteristics x 2 descriptions


def test_characteristic_that_does_not_fit_is_kept_warned_and_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.fit.DICT_PATH", tmp_path / "absent.json")
    repository = MemoryRepository()
    result = result_with("инструменты", "Противоскользящая рукоятка", "Стильный")
    with client(result, repository, Store()) as c:
        r = c.post("/analyze", json={"raw": "первое описание", "answers": []}).json()
        second = new_characteristics(c, "второе описание")
    assert [ch["name"] for ch in r["characteristics"] if ch["origin"] == "new"] == \
        ["Противоскользящая рукоятка", "Стильный"]  # nothing dropped
    warnings = [e for e in r["errors"] if e["type"] == "not_fitting_characteristic"]
    assert [(e["characteristic"], e["severity"]) for e in warnings] == [("Стильный", "warning")]
    assert [ch["known"] for ch in second] == [True, False]
    assert {key for _, _, key, _ in repository.rows} == {"противоскользящий рукоятка"}


def test_fit_check_works_without_postgres(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.fit.DICT_PATH", tmp_path / "absent.json")  # KTRU only: no power for tools
    with client(result_with("инструменты", "Мощность 800 Вт"), None, Store()) as c:
        r = c.post("/analyze", json={"raw": "молоток, мощность 800 Вт", "answers": []}).json()
    assert [e["characteristic"] for e in r["errors"] if e["type"] == "not_fitting_characteristic"] == \
        ["Мощность 800 Вт"]


class Classifier:
    """NewCharClassifier.predict contract: a fixed zone and P(valid) per characteristic name."""

    def __init__(self, zones: dict[str, tuple[str, float]]):
        self.zones = zones
        self.seen: list[str] = []

    def predict(self, items):
        self.seen += [it["name"] for it in items]
        return [self.zones[it["name"]] for it in items]


def test_classifier_judges_what_the_rules_let_through(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.fit.DICT_PATH", tmp_path / "absent.json")
    classifier = Classifier({"Противоскользящая рукоятка": ("valid", 0.97), "Размер обуви 42": ("invalid", 0.02),
                             "Влагостойкость": ("uncertain", 0.4)})
    monkeypatch.setattr("ml.service.analysis.AnalysisService._new_char_classifier", lambda self: classifier)
    repository = MemoryRepository()
    result = result_with("инструменты", "Противоскользящая рукоятка", "Размер обуви 42", "Влагостойкость", "Стильный")
    with client(result, repository, Store()) as c:
        r = c.post("/analyze", json={"raw": "молоток", "answers": []}).json()
    assert classifier.seen == ["Противоскользящая рукоятка", "Размер обуви 42", "Влагостойкость"]  # not the rules' one
    by_type = {(e["type"], e["characteristic"], e["checked_by"]) for e in r["errors"] if "checked_by" in e}
    assert by_type == {("not_fitting_characteristic", "Стильный", "rules"),
                       ("not_fitting_characteristic", "Размер обуви 42", "model"),
                       ("uncertain_characteristic", "Влагостойкость", "model")}
    assert len([ch for ch in r["characteristics"] if ch["origin"] == "new"]) == 4  # nothing dropped
    assert {key for _, _, key, _ in repository.rows} == {"противоскользящий рукоятка"}  # only the valid one


def test_portal_characteristic_absent_from_the_product_cards_is_warned(monkeypatch):
    class Cards:
        def portal_cards_without(self, category, product, name):
            return 120 if name in ("Разрешение экрана", "Вид инструмента") else None

        def predict(self, items):
            return [("valid", 0.9) for _ in items]

    monkeypatch.setattr("ml.service.analysis.AnalysisService._new_char_classifier", lambda self: Cards())
    result = result_with("инструменты")
    result["characteristics"] += [
        {"name": "Разрешение экрана", "value": "4K", "unit": None, "origin": "portal", "confidence": 0.8},
        {"name": "Вид инструмента", "value": "Молоток", "unit": None, "origin": "portal", "confidence": 0.8},
        {"name": "Цвет", "value": "Черный", "unit": None, "origin": "portal", "confidence": 0.8}]
    with client(result, None, Store()) as c:
        r = c.post("/analyze", json={"raw": "молоток", "answers": []}).json()
    # «Вид инструмента: Молоток» is the product itself, not a characteristic the user wrote
    assert [(e["characteristic"], e["checked_by"]) for e in r["errors"] if e.get("checked_by")] == \
        [("Разрешение экрана", "portal_cards")]


def test_unreachable_postgres_degrades(caplog):
    caplog.set_level(logging.INFO)
    with client(result_with("инструменты", "Противоскользящая рукоятка"),
                NewCharacteristicsRepository(UNREACHABLE_DSN)) as c:
        assert c.get("/health").json() == {"status": "degraded", "pipeline": "ready", "postgres": "unavailable"}
        assert c.get("/ready").status_code == 200  # Postgres is optional: the service stays ready
        for _ in range(2):  # the second request skips Postgres without waiting for a connection
            chars = new_characteristics(c, "молоток, противоскользящая рукоятка")
            assert chars and all("known" not in ch for ch in chars)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "step skipped" in r.getMessage()]
    assert len(warnings) == 1
    summaries = [r.getMessage() for r in caplog.records if r.getMessage().startswith("analyze: ")]
    assert [s.split("postgres=")[1].split()[0] for s in summaries] == ["failed", "skipped"]


@pytest.mark.skipif(not DSN, reason="TEST_POSTGRES_DSN is not set")
def test_characteristic_becomes_known_after_another_description():
    category = f"test-{uuid.uuid4().hex}"  # isolates the run from rows of earlier runs
    key = characteristic_key("Противоскользящая рукоятка")
    direct = NewCharacteristicsRepository(DSN)
    direct.pool.open(wait=True, timeout=10)
    try:
        with client(result_with(category, "Противоскользящая рукоятка", "Магнитный держатель"),
                    NewCharacteristicsRepository(DSN)) as c:
            assert c.get("/health").json() == {"status": "ok", "pipeline": "ready", "postgres": "ok"}
            first = new_characteristics(c, "молоток, противоскользящая рукоятка, магнитный держатель")
            assert [ch["known"] for ch in first] == [False, False]
            # the next round of the same dialog sends the same description: not a second sighting
            again = new_characteristics(c, "молоток, противоскользящая рукоятка, магнитный держатель")
            assert [ch["known"] for ch in again] == [False, False]
            other = new_characteristics(c, "молоток слесарный с противоскользящей рукояткой")
            assert [ch["known"] for ch in other] == [True, True]
        assert direct.seen_before(category, "стул", [key], "x") == {}  # another product has not seen it
        assert direct.seen_before(category, "молоток", [key], "y") == {key: 2}  # two distinct descriptions
    finally:
        with direct.pool.connection() as conn:
            conn.execute("DELETE FROM new_characteristics WHERE category = %s", (category,))
        direct.close()
