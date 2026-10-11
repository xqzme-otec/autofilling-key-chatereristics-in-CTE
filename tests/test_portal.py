import json

from ml.portal import PortalMatcher, card_index, clean_value, is_actual, new_characteristic, parse_card


def _matcher(tmp_path):
    path = tmp_path / "portal.json"
    path.write_text(json.dumps({"мебель": {
        "Цвет": {"usage": 50, "kind": "Справочник", "unit": None, "values": ["Синий", "Черный"]},
        "Цвет сиденья": {"usage": 5, "kind": "Справочник", "unit": None, "values": ["Синий"]},
        "Вес": {"usage": 30, "kind": "Дробное числовое значение", "unit": "кг", "values": []},
    }}, ensure_ascii=False), encoding="utf-8")
    return PortalMatcher(path)


def _profile_matcher(tmp_path):
    """The category reference plus the profile of one product, «молоток»."""
    def info(kind, usage=5, **extra):
        return {"usage": usage, "kind": kind, "unit": None, "values": [], "aliases": [], **extra}

    path = tmp_path / "portal.json"
    path.write_text(json.dumps({"инструменты": {
        "Материал": info("Справочник", usage=90, values=["Сталь", "Стеклопластик"]),
        "Материал корпуса": info("Справочник", usage=40, values=["Сталь"]),
        "Материал рукоятки": info("Справочник", values=["Дерево", "Стеклопластик"]),
        "Материал бойка": info("Текст"),
        "Обрезиненная ручка": info("Признак"),
        "Тип молотка": info("Текст", usage=1, local=1),
        "_profiles": {"молоток": {
            "cards": 863,
            "chars": {"Материал рукоятки": {"n": 561, "values": ["стеклопластик", "дерево"]},
                      "Материал бойка": {"n": 388, "values": ["сталь"]},
                      "Тип молотка": {"n": 336, "values": ["слесарный"]},
                      "Обрезиненная ручка": {"n": 198, "values": ["Да"], "flag": 1}},
            "by_value": {"стеклопластик": ["Материал рукоятки", "стеклопластик"],
                         "сталь": ["Материал бойка", "сталь"]}}},
    }}, ensure_ascii=False), encoding="utf-8")
    return PortalMatcher(path)


def test_portal_characteristic_by_name(tmp_path):
    m = _matcher(tmp_path)
    assert m.match("цвет сиденья синий", "мебель") == {"name": "Цвет сиденья", "value": "синий", "unit": None}
    assert m.match("вес 12", "мебель") == {"name": "Вес", "value": "12", "unit": "кг"}


def test_portal_characteristic_by_value(tmp_path):
    assert _matcher(tmp_path).match("синий", "мебель")["name"] == "Цвет"


def test_unknown_fragment_becomes_new_characteristic(tmp_path):
    assert _matcher(tmp_path).match("противоскользящая рукоятка", "мебель") is None
    assert new_characteristic("противоскользящая рукоятка") == {"name": "Противоскользящая рукоятка", "value": "Да", "unit": None}
    assert new_characteristic("длина кабеля 3 м")["name"] == "Длина кабеля"


def test_dictionary_without_profiles_ignores_the_product(tmp_path):
    m = _matcher(tmp_path)
    assert m.match("синий", "мебель", "стул") == m.match("синий", "мебель")
    assert m.has("мебель", "Цвет", "стул") and not m.has("мебель", "Материал рукоятки", "стул")


def test_value_goes_to_the_characteristic_of_this_product(tmp_path):
    m = _profile_matcher(tmp_path)
    # in the category «сталь» is first of all «Материал»; hammer cards give it to «Материал бойка»
    assert m.match("сталь", "инструменты")["name"] == "Материал"
    assert m.match("сталь", "инструменты", "молоток") == {"name": "Материал бойка", "value": "Сталь", "unit": None}
    assert m.match("стеклопластик", "инструменты", "молоток")["name"] == "Материал рукоятки"
    # a product without a profile falls back to the category
    assert m.match("сталь", "инструменты", "отвертка")["name"] == "Материал"


def test_yes_no_characteristic_of_the_product_named_as_is(tmp_path):
    m = _profile_matcher(tmp_path)
    assert m.match("обрезиненной ручкой", "инструменты", "молоток") == {"name": "Обрезиненная ручка", "value": "Да", "unit": None}
    assert m.match("обрезиненной ручкой", "инструменты") is None  # outside the product only «name value» works


def test_characteristic_known_only_from_product_cards(tmp_path):
    m = _profile_matcher(tmp_path)
    assert m.match("тип молотка слесарный", "инструменты", "молоток") == {"name": "Тип молотка", "value": "слесарный", "unit": None}
    assert m.match("тип молотка слесарный", "инструменты") is None
    assert m.has("инструменты", "Тип молотка", "молоток") and not m.has("инструменты", "Тип молотка")


def test_card_characteristics_are_split_by_reference_names():
    index = card_index(["Тип", "Тип патрона", "Вес", "Высота упаковки", "Мощность"])
    raw = "Вес 2.03000 кг | Высота упаковки 100 мм | Тип патрона быстрозажимной | Мощность 0.00000 Вт | " \
          "Тип (не задано) | Цвет синий | Вес 5 кг"
    # zero and empty values, an unknown name and a repeated name are skipped
    assert parse_card(raw, index) == [("Вес", "2.03 кг"), ("Высота упаковки", "100 мм"), ("Тип патрона", "быстрозажимной")]


def test_clean_value():
    assert clean_value(" 500.00000 г ") == "500 г"
    assert clean_value("0.50000 кг") == "0.5 кг"
    assert clean_value("1.05") == "1.05"
    assert clean_value("0.00000") == "" and clean_value("(не задано) мм") == "" and clean_value("  ") == ""


def test_only_actual_characteristics():
    assert is_actual({"Эталонная"}, 10) and is_actual({"Базовая", "Эталонная"}, 1)
    assert not is_actual({"Базовая"}, 0)  # in the reference, but no card fills it in
    assert not is_actual({"Базовая", "Категореобразующая"}, 500)  # service: «Вид продукции: Товары»
    assert not is_actual({"?"}, 500)
