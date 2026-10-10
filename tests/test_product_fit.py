"""Does a new characteristic fit the product: ml/fit.py on a small KTRU store and portal dictionary."""
from __future__ import annotations

import json

import pytest

from ml.fit import ProductFit, unit_dimension


def position(code: str, name: str, category: str, *units: str) -> dict:
    return {"code": code, "name": name, "category": category,
            "characteristics": [{"name": f"Характеристика {i}", "unit": u, "values": []} for i, u in enumerate(units)]}


class Store:
    items = [position(f"t{i}", "Дрель", "инструменты", "Ватт") for i in range(3)] + [
        position("t9", "Перфоратор", "инструменты", "Миллиметр"),
        position("f1", "Стол письменный", "мебель", "Миллиметр"),
        position("c1", "Жакет женский", "одежда", "Сантиметр"),
    ]


@pytest.fixture(scope="module")
def fit(tmp_path_factory) -> ProductFit:
    portal = {"инструменты": {"Энергия удара": {"usage": 9, "kind": "Числовая", "unit": "Дж", "values": []}},
              "мебель": {"Объем": {"usage": 10, "kind": "Числовая", "unit": "Л; ДМ3", "values": []}},
              "одежда": {}}
    path = tmp_path_factory.mktemp("portal") / "portal.json"
    path.write_text(json.dumps(portal, ensure_ascii=False), encoding="utf-8")
    return ProductFit(Store(), path)


@pytest.mark.parametrize("category, product, name, value", [
    ("мебель", "стол", "Объем двигателя", "2 л"),             # volume: rare portal usage, no KTRU
    ("мебель", "стол", "Объем двигателя 2", "Да"),            # no unit: the name tells the dimension
    ("одежда", "жакет", "Мощность", "800 Вт"),
    ("одежда", "жакет", "Емкость аккумулятора 2000 мАч", "Да"),
    ("инструменты", "перфоратор", "Калорийность", "200 ккал"),
    ("одежда", "жакет", "Стильный", "Да"),                    # subjective
    ("одежда", "жакет", "Отличное качество", "Да"),
])
def test_does_not_fit(fit, category, product, name, value):
    assert fit.reason(category, product, name, value)


@pytest.mark.parametrize("category, product, name, value", [
    ("инструменты", "дрель", "Мощность", "800 Вт"),           # KTRU positions of the category have power
    ("инструменты", "перфоратор", "Максимальная энергия удара", "1.9 Дж"),  # multi-word portal name, same dimension
    ("мебель", "стол", "Ширина", "120 см"),                   # size fits any product
    ("одежда", "жакет", "Объем груди", "96 см"),              # the unit wins over the name
    ("мебель", "стол", "Высота 75", "Да"),
    ("инструменты", "молоток", "Противоскользящая рукоятка", "Да"),
    ("одежда", "жакет", "Очень", "Да"),                       # an intensifier alone is not an evaluation
])
def test_fits(fit, category, product, name, value):
    assert fit.reason(category, product, name, value) is None


def test_unit_dimension():
    assert [unit_dimension(u) for u in ("л", "литров", "В", "вольт", "мАч", "об/мин", "кВт", "Дж", "бар", "высота")] == \
        ["volume", "volume", "voltage", "voltage", "charge", "rpm", "power", "energy", "pressure", None]
