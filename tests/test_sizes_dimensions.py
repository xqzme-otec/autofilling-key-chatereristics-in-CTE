import json

import pytest

from ml.ktru import KtruStore
from ml.sizes import SizeMatcher, canonical_letter
from ml.text import normalize
from ml.units import NumberMatcher, find_dimensions
from ml.values import ValueMatcher


def _num(name, unit, bounds):
    """A numeric characteristic with consecutive ranges [a, b) and open ends."""
    values = [{"max": bounds[0], "max_incl": False}]
    values += [{"min": a, "min_incl": True, "max": b, "max_incl": False} for a, b in zip(bounds, bounds[1:])]
    values.append({"min": bounds[-1], "min_incl": True})
    return {"name": name, "required": True, "unit": unit, "values": values}


def _text(name, values):
    return {"name": name, "required": True, "unit": None, "values": [{"text": v} for v in values]}


ITEMS = [
    {"code": "CAB", "name": "Шкаф для одежды металлический", "category": "мебель", "characteristics": [
        _num("Длина", "Миллиметр", [500, 600, 700, 800, 900, 1000, 1200, 1500]),
        _num("Высота", "Миллиметр", [1000, 1200, 1400, 1600, 1800, 1900, 2000, 2100, 2200]),
        _num("Глубина", "Миллиметр", [400, 500, 600]),
        _num("Длина полки", "Миллиметр", [300, 400, 500, 600])]},
    {"code": "TABLE", "name": "Стол обеденный", "category": "мебель", "characteristics": [
        _num("Длина столешницы", "Миллиметр", [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]),
        _num("Ширина столешницы", "Миллиметр", [500, 600, 700, 800, 900, 1000]),
        _num("Высота до столешницы", "Миллиметр", [700, 750, 800, 850, 900, 1000])]},
    {"code": "DESK", "name": "Стол письменный", "category": "мебель", "characteristics": [
        {"name": "Количество выдвижных ящиков", "required": False, "unit": "Штука", "values": [{"min": 0, "max": 0}]}]},
    {"code": "RACK", "name": "Стеллаж офисный", "category": "мебель", "characteristics": [
        _num("Высота стеллажа", "Миллиметр", list(range(1200, 2700, 100))),
        _num("Ширина стеллажа", "Миллиметр", list(range(400, 2100, 100))),
        _num("Глубина стеллажа", "Миллиметр", [300, 350, 400, 450, 500, 550, 600, 650, 700, 1000]),
        _num("Высота шрифта", "Миллиметр", [1, 2, 3])]},
    {"code": "JACKET", "name": "Куртка зимняя", "category": "одежда", "characteristics": [
        {"name": "Российский размер", "required": True, "unit": None,
         "values": [{"min": n, "max": n} for n in range(44, 62, 2)]}]},
    {"code": "JEANS", "name": "Джинсы", "category": "одежда", "characteristics": [
        _text("Российский размер", ["44", "44-46", "46", "48", "50", "52", "52-54", "54", "56"])]},
    {"code": "APRON", "name": "Фартук для защиты от излучения", "category": "одежда", "characteristics": [
        _text("Размер", ["S", "M", "L", "XL", "XXL"])]},
    {"code": "BIB", "name": "Нагрудник для защиты от излучения", "category": "одежда", "characteristics": [
        _text("Размер", ["М"])]},  # cyrillic «М», as in the real KTRU
    {"code": "KEY", "name": "Ключ гаечный", "category": "инструменты", "characteristics": [
        _num("Размер зева", "Миллиметр", [8, 10, 12, 14])]},
]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("ktru") / "ktru.json"
    path.write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    return KtruStore(path)


def _dims(text):
    t = normalize(text)
    return [(t[g["start"]:g["end"]], g["nums"], [u[1] for u in g["units"]], g["labels"]) for g in find_dimensions(t)]


# ---------- letter sizes ----------

def test_canonical_letter():
    assert [canonical_letter(x) for x in ["xl", "ХL", "2XL", "3xl", "XXXXL", "М", "s", "XM", "5"]] == \
        ["XL", "XL", "XXL", "XXXL", "4XL", "M", "S", None, None]


def test_letter_size_mapped_to_russian_size(store):
    found, errors, _, _ = SizeMatcher(store).match("куртка зимняя мужская размер XL", ["JACKET"])
    assert [(f["name"], f["value"], f["method"], f["codes"]) for f in found] == \
        [("Российский размер", "52–54", "size_table", ["JACKET"])]
    assert errors == []


def test_women_table_and_letter_range(store):
    m = SizeMatcher(store)
    assert m.match("куртка женская, XL", ["JACKET"])[0][0]["value"] == "48–50"
    assert m.match("куртка зимняя S-M", ["JACKET"])[0][0]["value"] == "46–50"


def test_letter_size_matched_directly(store):
    m = SizeMatcher(store)
    found = m.match("фартук для защиты от излучения, размер 2XL", ["APRON"])[0]
    assert [(f["name"], f["value"], f["method"]) for f in found] == [("Размер", "XXL", "exact")]
    assert m.match("нагрудник для защиты от излучения M", ["BIB"])[0][0]["value"] == "М"  # KTRU's cyrillic value
    assert m.match("фартук для защиты, двухсторонний, l", ["APRON"])[0][0]["value"] == "L"  # a whole fragment


def test_letter_size_not_in_ktru_is_an_error(store):
    found, errors, _, _ = SizeMatcher(store).match("фартук для защиты от излучения, XS", ["APRON"])
    assert found == [] and [e["type"] for e in errors] == ["nonexistent_value"]


def test_metres_and_models_are_not_sizes(store):
    m = SizeMatcher(store)
    for text in ["фартук 2 м", "фартук 2 M", "фартук M 12", "фартук L-образный", "фартук для защиты l"]:
        assert m.match(text, ["APRON"])[0] == [], text
    # «2 м» is not «Размер: М» on the text-value level either
    assert ValueMatcher(store).match("нагрудник 2 м", ["BIB"])[0] == []


def test_size_without_ktru_characteristic_is_kept(store):
    found, errors, _, unmatched = SizeMatcher(store).match("куртка детская XL", ["JACKET"])
    assert found == [] and errors == []
    assert [(u["keyword"], u["value_text"]) for u in unmatched] == [("Размер", "XL")]


def test_russian_size_after_size_word(store):
    m = SizeMatcher(store)
    assert m.match("джинсы р. 52-54", ["JEANS"])[0][0]["value"] == "52-54"      # the KTRU value itself
    assert m.match("куртка р-р 52-54", ["JACKET"])[0][0]["value"] == "52–54"    # both 52 and 54 fit
    found, errors, _, _ = m.match("куртка, р. 62", ["JACKET"])  # sizes here end at 60
    assert found == [] and errors[0]["type"] == "nonexistent_value"
    assert m.match("куртка р. 26", ["JACKET"])[:2] == ([], [])   # not an adult size: left to the other matchers


def test_size_by_chest_girth(store):
    # workwear «р. 96-100/182-188»: chest girth 96–100 cm / height; Russian size = half the chest girth
    found = SizeMatcher(store).match("куртка р. 96-100/182-188", ["JACKET"])[0]
    assert [(f["value"], f["method"]) for f in found] == [("48–50", "size_table")]


def test_letter_with_russian_size(store):
    assert SizeMatcher(store).match("куртка р.L/48-50", ["JACKET"])[0][0]["value"] == "48–50"


# ---------- dimensions ----------

def test_dimension_formats():
    assert _dims("стол 120х60х75 см") == [("120х60х75 см", [120, 60, 75], [10, 10, 10], [None] * 3)]
    assert _dims("шкаф 1800x500x600 мм")[0][2] == [1, 1, 1]
    assert _dims("стол 1200х600х750")[0][2] == [1, 1, 1]           # no unit, > 300 → millimetres
    assert _dims("стол 120*60*75")[0][2] == [10, 10, 10]           # no unit, ≤ 300 → centimetres
    assert _dims("матрас 200 на 90 см")[0][1] == [200, 90]
    assert _dims("стол 120×60 см")[0][1] == [120, 60]
    assert _dims("шкаф Ш120 Г60 В75")[0][3] == ["Ширина", "Глубина", "Высота"]
    assert _dims("кровать д190*ш71*в46")[0][3] == ["Длина", "Ширина", "Высота"]
    assert _dims("шкаф, ШхГхВ 500х600х1800 мм")[0][3] == ["Ширина", "Глубина", "Высота"]
    assert _dims("габариты (ДхШхВ), мм: 1200х60х75")[0][2] == [1, 1, 1]  # the unit is in the header


def test_not_dimensions():
    for text in ["кабель 3х2.5 мм2", "лампа 2x20 Вт", "болт M8x1.25", "1x2x3x4", "2 м", "размер 52-54", "в 2 слоя",
                 "дрель 220 В", "3 полки на 50 кг"]:
        assert _dims(text) == [], text


def _assigned(store, text, codes):
    found, errors, reserved, unmatched = NumberMatcher(store).match(normalize(text), codes)
    return {f["name"]: f["value"] for f in found}, errors, unmatched


def test_dimensions_mapped_to_ktru(store):
    chars, errors, _ = _assigned(store, "стол обеденный 1200x800x750", ["TABLE"])
    assert chars == {"Длина столешницы": "1200–1300", "Ширина столешницы": "800–900", "Высота до столешницы": "750–800"}
    assert errors == []


def test_order_chosen_by_ktru_ranges(store):
    # «В×Ш×Г»: the height is 2000, not the first number by convention; «Высота шрифта» is a part, not the rack
    chars, _, _ = _assigned(store, "стеллаж 2000*1000*400", ["RACK"])
    assert chars == {"Высота стеллажа": "2000–2100", "Ширина стеллажа": "1000–1100", "Глубина стеллажа": "400–450"}
    chars, _, _ = _assigned(store, "шкаф 1800x500x600 мм", ["CAB"])
    assert chars["Высота"] == "1800–1900"


def test_labelled_axes(store):
    # «Ш» of a cabinet that has «Длина × Глубина × Высота» is its «Длина»
    chars, _, _ = _assigned(store, "шкаф Ш500 Г600 В1800", ["CAB"])
    assert chars == {"Длина": "500–600", "Глубина": "≥ 600", "Высота": "1800–1900"}


def test_dimensions_without_ktru_place_are_kept(store):
    chars, errors, unmatched = _assigned(store, "стол письменный 120х60х75 см", ["DESK"])
    assert chars == {} and errors == []
    assert [(u["keyword"], u["value_text"]) for u in unmatched] == [("Длина", "120 см"), ("Ширина", "60 см"),
                                                                    ("Высота", "75 см")]


def test_dimension_errors(store):
    _, errors, _ = _assigned(store, "шкаф 18000x500x600 мм", ["CAB"])
    assert [e["type"] for e in errors] == ["suspicious_value"]
    assert "18000 мм" in errors[0]["message"]


def test_bare_pair_is_not_a_size_for_tools(store):
    chars, errors, unmatched = _assigned(store, "ключ гаечный 10х12", ["KEY"])
    assert chars == {} and errors == [] and unmatched == []


def test_packaging_and_part_sizes_are_kept_whole(store):
    chars, errors, unmatched = _assigned(store, "стол обеденный, габариты в упаковке 120x80x10 см", ["TABLE"])
    assert chars == {} and errors == []
    assert [(u["keyword"], u["value_text"]) for u in unmatched] == [("габариты в упаковке", "120x80x10 см")]
    # the first group is the product's, a later one after other words is a part's
    chars, _, unmatched = _assigned(store, "стол обеденный 1200x800x750, металлический уголок 140x120 мм", ["TABLE"])
    assert len(chars) == 3 and [u["keyword"] for u in unmatched] == ["металлический уголок"]
    chars, _, _ = _assigned(store, "стол обеденный, дуб светлый 1200x800x750", ["TABLE"])
    assert len(chars) == 3


def test_dimension_name_is_reserved(store):
    text = normalize("стол обеденный, габаритные размеры 1200x800x750 мм")
    found, _, reserved, _ = NumberMatcher(store).match(text, ["TABLE"])
    assert len(found) == 3
    assert {text[s:e] for s, e in reserved} >= {"габаритные", "размеры"}


def test_children_girth_is_not_converted(store):
    assert SizeMatcher(store).match("куртка для мальчика р. 146", ["JACKET"])[:2] == ([], [])
