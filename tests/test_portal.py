import json

from ml.portal import PortalMatcher, new_characteristic


def _matcher(tmp_path):
    path = tmp_path / "portal.json"
    path.write_text(json.dumps({"мебель": {
        "Цвет": {"usage": 50, "kind": "Справочник", "unit": None, "values": ["Синий", "Черный"]},
        "Цвет сиденья": {"usage": 5, "kind": "Справочник", "unit": None, "values": ["Синий"]},
        "Вес": {"usage": 30, "kind": "Дробное числовое значение", "unit": "кг", "values": []},
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
