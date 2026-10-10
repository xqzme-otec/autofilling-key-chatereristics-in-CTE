from pathlib import Path

import pytest

from ml.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data/processed/ktru.parquet").exists()
    or not (ROOT / "data/processed/portal_characteristics.json").exists(),
    reason="real KTRU and supplier portal artifacts are required",
)


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


def _has_color(result, value):
    return any(c["name"] == "Цвет" and c["value"] == value and c["source"] == "portal"
               for c in result["characteristics"])


def test_misspelled_dress(pipeline):
    result = pipeline.analyze("плате чернные")
    assert result["text"].startswith("платье")
    assert result["product_intent"] == "платье"
    assert _has_color(result, "Черный")
    assert result["next_step"]["type"] == "product"
    assert all("Платье" in c["name"] for c in result["candidates"])


def test_black_womens_dress(pipeline):
    result = pipeline.analyze("платье черное женское")
    assert result["product_intent"] == "платье"
    assert _has_color(result, "Черный")
    assert any(s["value"] == "Женское" for s in result["subtype_context"])
    assert all("Платье" in c["name"] for c in result["candidates"])


def test_kitchen_stone_table_and_answer(pipeline):
    result = pipeline.analyze("стол кузонный каменный")
    assert result["text"].startswith("стол кухонный")
    assert result["product_intent"] == "стол"
    assert any(s["value"] == "Кухонный" and s["used_for_reranking"] for s in result["subtype_context"])
    assert any(c["name"] == "Вид материала столешницы" and c["value"] == "Камень"
               and c["used_for_narrowing"] for c in result["characteristics"])
    assert result["candidates"][0]["name"] == "Стол обеденный"
    assert result["next_step"]["type"] == "product"
    assert result["next_step"]["options"][0] == "Стол обеденный"
    answered = pipeline.analyze({"raw": "стол кузонный каменный",
                                 "answers": [{"type": "product", "value": "Стол обеденный"}]})
    assert answered["remaining_count"] < result["remaining_count"]
    assert all(c["name"] == "Стол обеденный" for c in answered["candidates"])
    assert answered["ktru"] or answered["next_step"]


def test_written_wooden_table(pipeline):
    result = pipeline.analyze("стол письменный деревянный")
    assert result["product_intent"] == "стол"
    assert any(s["value"] == "Письменный" for s in result["subtype_context"])
    assert all(c["name"] == "Стол письменный" for c in result["candidates"])


def test_locksmith_hammer(pipeline):
    result = pipeline.analyze("молоток слесарный 600 грамм ручка 25 см")
    assert result["product_intent"] == "молоток"
    assert any(c["value"] == "Слесарный" for c in result["characteristics"])
    assert all(c["name"] == "Молоток" for c in result["candidates"])
    assert not any(c["value"] == "Да" and c["name"] in ("Ручка", "Тяжелый") for c in result["characteristics"])


def test_white_medical_coat(pipeline):
    result = pipeline.analyze("халат медицинский белый")
    assert result["product_intent"] == "халат"
    assert any(s["value"] == "Медицинский" for s in result["subtype_context"])
    assert _has_color(result, "Белый")
    assert "производственный" not in result["candidates"][0]["name"].lower()
    assert result["next_step"]
