"""What is not a characteristic: brands and models, gibberish, the product itself, a name without a value,
«по металлу»; and dimensions judged by the product, not by its whole category. Real KTRU and portal data."""
from pathlib import Path

import pytest

from ml.pipeline import _brand_word, _gibberish, _trigrams

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "data/processed/ktru.parquet").exists() and (ROOT / "data/processed/portal_characteristics.json").exists()
needs_data = pytest.mark.skipif(not DATA, reason="real KTRU and supplier portal artifacts are required")


@pytest.fixture(scope="module")
def pipeline():
    from ml.pipeline import Pipeline
    return Pipeline()


def extra(result):
    return {(c["name"], c["value"]) for c in result["characteristics"] if c["origin"] in ("portal", "new")}


def notices(result, kind):
    return [e for e in result["errors"] if e["type"] == kind]


def test_gibberish_is_told_from_rare_words():
    # a stand-in for the KTRU vocabulary: real words share letter triples with rare real words
    trigrams = _trigrams(["ширина", "гвардия", "гарнир", "нерест", "перила", "однослойный", "слойка", "оная"])
    for word in ("ыврпаоы", "ппп", "зщкклм", "ваываыв", "кгвыап"):
        assert _gibberish(word, trigrams, set()), word
    for word in ("ширина", "гварнери", "однослойная"):
        assert not _gibberish(word, trigrams, set()), word


def test_brand_words():
    text = "кресло руководителя Юнитекс, Цвет черный, ШХА-900"
    vocab = {"цвет", "черный", "кресло"}
    assert _brand_word("Юнитекс", text, text.index("Юнитекс"), vocab)
    assert not _brand_word("Цвет", text, text.index("Цвет"), vocab)  # starts a clause
    assert _brand_word("ШХА", text, text.index("ШХА"), vocab)
    assert _brand_word("Bosch", "дрель Bosch", 6, vocab)
    assert not _brand_word("Полипропилен", "Материал Полипропилен", 9, {"полипропилен"})  # a value


@needs_data
@pytest.mark.parametrize("text, brand", [
    ("дрель Bosch 750 Вт", "Bosch"),
    ("кресло руководителя Юнитекс кожа черное", "Юнитекс"),
    ("брюки женские «Стелла» серые", "Стелла"),
    ("кресло офисное Бюрократ CH-868", "CH-868"),
])
def test_brands_are_reported_not_turned_into_characteristics(pipeline, text, brand):
    r = pipeline.analyze(text)
    assert not any(brand.lower() in name.lower() for name, _ in extra(r))
    assert any(brand in n for e in notices(r, "brand_or_model") for n in e["characteristics"])


@needs_data
@pytest.mark.parametrize("text, fragment", [
    ("стол письменный, ыврпаоы", "ыврпаоы"),
    ("стол письменный, ппп ффф ыыы", "ппп ффф ыыы"),
    ("куртка мужская, зщкклм 48", "зщкклм 48"),
])
def test_gibberish_asks_to_clarify(pipeline, text, fragment):
    r = pipeline.analyze(text)
    assert [e["source"] for e in notices(r, "unreadable_fragment")] == [fragment]
    assert not any(fragment.split()[0] in name.lower() for name, _ in extra(r))


@needs_data
def test_purpose_is_not_a_material(pipeline):
    r = pipeline.analyze("сверло по металлу 6 мм")
    assert not any(name == "Металл" for name, _ in extra(r))
    assert any("металл" in value.lower() for _, value in extra(r))


@needs_data
def test_name_with_a_code_value(pipeline):
    r = pipeline.analyze("сверло материал изготовления HSS")
    assert ("Материал изготовления", "HSS") in extra(r)
    assert not notices(r, "brand_or_model")


@needs_data
def test_name_without_a_value_asks_for_it(pipeline):
    r = pipeline.analyze("шкаф для документов, материал изготовления")
    assert [e["characteristic"] for e in notices(r, "missing_value")] == ["Материал изготовления"]
    assert not any(value == "Да" and "материал" in name.lower() for name, value in extra(r))


@needs_data
def test_the_product_itself_is_not_a_characteristic(pipeline):
    r = pipeline.analyze("куртка женская размер M")
    assert not any(value.lower().startswith("куртка") for _, value in extra(r))


@needs_data
def test_a_name_inside_a_clause_is_not_a_typo(pipeline):
    r = pipeline.analyze("Стол офисный Модерат 1200х600")
    assert "Модерат" in r["text"]
    assert not any(c["from"] == "Модерат" for c in r["corrections"])


@needs_data
def test_dimensions_are_judged_by_the_product():
    from ml.fit import ProductFit
    from ml.ktru import KtruStore
    fit = ProductFit(KtruStore())
    assert fit.reason("инструменты", "молоток", "Объем двигателя", "2 л")  # chainsaws have it, hammers do not
    assert fit.reason("инструменты", "дрель", "Мощность", "750 Вт") is None
    assert fit.reason("инструменты", "шуруповерт", "Напряжение", "18 В") is None


@needs_data
@pytest.mark.parametrize("text, name, value", [
    ("футболка мужская 2 литра", "Объем", "2 литра"),          # volume is a unit: the number is not lost
    ("стол письменный, длина рукава 60 см", "Длина рукава", "60 см"),  # the whole name, not «Рукава» + «Длина: Да»
    ("молоток слесарный, частота вращения 3000 об/мин", "Частота вращения", "3000 об/мин"),
    ("шкаф, в собранном виде Нет", "Собранный вид", "Нет"),     # the answer after the phrase is its value
])
def test_names_and_values_are_kept_whole(pipeline, text, name, value):
    assert (name, value) in extra(pipeline.analyze(text))


@needs_data
def test_a_part_the_product_does_not_have_is_not_its_dimension(pipeline):
    r = pipeline.analyze("стеллаж металлический, длина рукава 60 см")
    assert ("Длина рукава", "60 см") in extra(r)
    assert not any(c["origin"] == "ktru" and "60" in c.get("source", "") for c in r["characteristics"])


@needs_data
def test_words_of_another_category():
    from ml.fit import ProductFit
    from ml.ktru import KtruStore
    fit = ProductFit(KtruStore())
    assert fit.reason("мебель", "витрина", "Тип изделия", "пассатижи")       # a tool named as a furniture type
    assert fit.reason("одежда", "брюки", "Цвет", "лдсп серый")
    assert fit.reason("одежда", "куртка", "Подогрев", "Да") is None          # rare for clothing: ask, not reject
    assert fit.doubt("одежда", "куртка", "Подогрев", "Да")
    for category, product, name in (("одежда", "куртка", "Светоотражающие полосы"),
                                    ("инструменты", "молоток", "Обрезиненная рукоятка"),
                                    ("мебель", "кресло", "Механизм качания"), ("мебель", "кресло", "Функция массажа")):
        assert fit.reason(category, product, name, "Да") is None and fit.doubt(category, product, name, "Да") is None


@needs_data
def test_invalid_characteristics_are_not_added():
    from fastapi.testclient import TestClient
    from ml.service.app import create_app
    from ml.service.config import Settings
    with TestClient(create_app(Settings(), load_in_background=False)) as c:
        r = c.post("/analyze", json={"raw": "футболка мужская 2 литра", "answers": []}).json()
    assert not any("объем" in ch["name"].lower() or "литр" in ch["name"].lower() for ch in r["characteristics"])
    assert any(e["type"] == "not_fitting_characteristic" for e in r["errors"])
