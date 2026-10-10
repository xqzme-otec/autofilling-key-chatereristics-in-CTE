import pytest

from ml.ktru import MOCK_PATH, KtruStore
from ml.units import NumberMatcher
from ml.values import ValueMatcher

HAMMER = "25.73.30.141-MOCK0006"


@pytest.fixture(scope="module")
def store():
    return KtruStore(MOCK_PATH)


def test_number_mapped_to_ktru_range(store):
    found, errors, _ = NumberMatcher(store).match("молоток массой 0.5 кг", [HAMMER])
    assert [(f["name"], f["value"], f["unit"]) for f in found] == [("Масса молотка", "0.4–0.8", "Килограмм")]
    assert errors == []


def test_suspicious_value_in_open_range(store):
    found, errors, _ = NumberMatcher(store).match("молоток массой 50 кг", [HAMMER])
    assert found[0]["value"] == "≥ 2.5"
    assert [(e["type"], e["severity"]) for e in errors] == [("suspicious_value", "warning")]


def test_unit_mismatch(store):
    found, errors, _ = NumberMatcher(store).match("молоток, длина рукоятки 2 кг", [HAMMER])
    assert found == []
    assert [e["type"] for e in errors] == ["unit_mismatch"]


def test_article_number_is_not_a_value(store):
    found, errors, _ = NumberMatcher(store).match("молоток STAYER 202815 массой 0.5 кг", [HAMMER])
    assert [f["source"] for f in found] == ["0.5 кг"]
    assert errors == []


def test_number_is_reserved_for_text_matching(store):
    _, _, reserved = NumberMatcher(store).match("молоток массой 0.5 кг", [HAMMER])
    text = "молоток массой 0.5 кг"
    assert any(text[s:e] == "0.5" for s, e in reserved)


def test_nonexistent_value_named_characteristic(store):
    chars, errors, _ = ValueMatcher(store).match("стул офисный, обивка алмазная", ["31.01.11.150-MOCK0001"])
    assert chars == []
    assert errors[0]["type"] == "nonexistent_value"
    assert errors[0]["characteristic"] == "Материал обивки"


def test_other_attribute_is_not_an_error(store):
    # «цвет обивки» — другая характеристика, а не значение «Материал обивки»
    _, errors, _ = ValueMatcher(store).match("стул офисный, цвет обивки синий", ["31.01.11.150-MOCK0001"])
    assert errors == []
