"""Does a new characteristic (origin "new": in neither KTRU nor the portal dictionary) fit the product?

Two rules, both chosen on a benchmark of real portal card characteristics and hand-labeled fragments; what they
let through is judged by the classifier of ml/newchar.py:

  1. subjective — only evaluative words: «классная», «стильный», «отличного качества»;
  2. foreign dimension — the unit (or, without a unit, the name) means a dimension that products of this
     category never have: «объём двигателя 2 л» for a table, «мощность 800 Вт» for a T-shirt. Size, weight,
     quantity, share, temperature and time fit any product. A dimension fits a category when its KTRU
     positions or its portal characteristics (sheet 2 of the exports) use it often enough; a rarer one
     fits when the name contains a multi-word portal characteristic of that dimension: «Максимальная
     энергия удара 1.9 Дж» for a perforator (portal: «Энергия удара», Дж).

A characteristic that does not fit is kept in the result (docs/DOC.md, invariant 2): it is only not
remembered, and the user gets a warning.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.portal import DICT_PATH
from ml.text import main_noun, tokenize

logger = logging.getLogger(__name__)

# a dimension fits a category when this many KTRU positions have it, or its portal characteristics are used in
# this many KPGZ categories in total (furniture: 30 for volume, from a few water coolers — not enough)
MIN_KTRU_POSITIONS = 3
MIN_PORTAL_USAGE = 60
UNIVERSAL_DIMS = {"length", "mass", "count", "share", "temp", "time", "area", "density"}

# unit as written by users and in the portal exports → dimension: exact abbreviations and word stems
# («литров», «вольта»); checked in order, first match wins
UNIT_DIMS = [
    ("charge", r"а\.?ч|а\*ч|ач|мач|ма·ч|а·ч", ("ампер-час",)),
    ("rpm", r"об/мин|об\.?/мин|оборотов в минуту.*", ("оборот",)),
    ("speed", r"км/ч|м/с", ()),
    ("power", r"вт|квт|л\.\s*с|лс", ("ватт", "киловатт", "лошадин")),
    ("voltage", r"в", ("вольт",)),
    ("volume", r"мл|л|дм3|дм\[3\*\]|см\[3\*\].*|л; дм3", ("литр", "миллилитр")),
    ("pressure", r"бар|па|кпа|мпа|атм", ("атмосфер", "паскал")),
    ("energy", r"дж|кдж|кал|ккал", ("джоул", "калори", "килокалори")),
    ("frequency", r"гц|кгц", ("герц",)),
    ("length", r"мм|см|м|дм|км|\"", ("миллиметр", "сантиметр", "метр", "дюйм")),
    ("mass", r"г|гр|кг|т", ("грамм", "килограмм", "тонн", "кило")),
    ("count", r"шт", ("штук",)),
]
UNIT_RES = [(dim, re.compile(rf"^(?:{abbr})$", re.IGNORECASE), stems) for dim, abbr, stems in UNIT_DIMS]
KTRU_UNIT_DIMS = {"Ватт": "power", "Киловатт": "power", "Вольт": "voltage", "Ампер-час (3,6 кКл)": "charge",
                  "Оборот в минуту": "rpm"}
# without a unit the name tells the dimension: «объём двигателя 2», «калорийность 200»
NAME_DIMS = {"объем": "volume", "литраж": "volume", "мощность": "power", "напряжение": "voltage",
             "аккумулятор": "charge", "вращение": "rpm", "оборот": "rpm", "давление": "pressure",
             "калорийность": "energy", "энергия": "energy"}
NUMBER_UNIT = re.compile(r"\d+(?:[.,]\d+)?\s*([^\s\d,;()]+(?:\s*/\s*[^\s\d,;()]+)?)", re.IGNORECASE)

SUBJECTIVE = set("""классный удобный модный красивый стильный крутой отличный лучший прекрасный шикарный
элегантный практичный комфортный популярный оригинальный уникальный идеальный превосходный качественный
великолепный замечательный хороший клевый симпатичный милый трендовый эксклюзивный роскошный изысканный
восхитительный надежный любимый офигенный суперский супер топовый престижный незаменимый премиальный вечный
качество очень""".split())
SUBJECTIVE_ONLY = {"качество", "очень"}  # count as evaluative only next to an evaluative word


def _words(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalpha()]


def unit_dimension(unit: str) -> str | None:
    unit = unit.strip().lower().replace("ё", "е").rstrip(".")
    return next((dim for dim, rx, stems in UNIT_RES if rx.match(unit) or unit.startswith(stems or ("\0",))), None)


class ProductFit:
    def __init__(self, store, portal_path: Path | None = None):
        portal_path = portal_path or DICT_PATH
        positions: dict[tuple[str, str], set] = defaultdict(set)   # (category, product) -> dimensions
        category_count: dict[str, Counter] = defaultdict(Counter)  # category -> dimension -> positions
        for item in store.items:
            dims = {KTRU_UNIT_DIMS[ch["unit"]] for ch in item["characteristics"] if ch.get("unit") in KTRU_UNIT_DIMS}
            positions[(item["category"], main_noun(item["name"]))] |= dims
            category_count[item["category"]].update(dims)
        self.product_dims = dict(positions)
        self.category_dims = {c: {d for d, n in cnt.items() if n >= MIN_KTRU_POSITIONS} for c, cnt in category_count.items()}
        self.portal_names: dict[str, list[tuple[frozenset, str]]] = defaultdict(list)  # category -> (lemmas, dim)
        if portal_path.exists():
            for category, chars in json.loads(portal_path.read_text(encoding="utf-8")).items():
                usage: Counter = Counter()
                for name, info in chars.items():
                    dim = unit_dimension(info["unit"]) if info.get("unit") else None
                    if dim:
                        usage[dim] += info["usage"]
                        lemmas = frozenset(_words(name))
                        if len(lemmas) >= 2:
                            self.portal_names[category].append((lemmas, dim))
                self.category_dims.setdefault(category, set()).update(d for d, n in usage.items() if n >= MIN_PORTAL_USAGE)
        logger.info("product fit: dimensions per category %s",
                    {c: sorted(d - UNIVERSAL_DIMS) for c, d in self.category_dims.items()})

    def reason(self, category: str, product: str, name: str, value: str) -> str | None:
        """Why the characteristic does not fit the product, or None when it fits."""
        text = name if value in ("", "Да", "Нет") else f"{name} {value}"
        words = _words(text)
        if words and all(w in SUBJECTIVE for w in words) and any(w not in SUBJECTIVE_ONLY for w in words):
            return f"«{text}» — субъективная оценка, а не характеристика товара"
        dim = self._dimension(text, words)
        if dim and dim not in UNIVERSAL_DIMS and dim not in self.category_dims.get(category, set()) \
                and dim not in self.product_dims.get((category, product), set()) \
                and not any(d == dim and lemmas <= set(words) for lemmas, d in self.portal_names.get(category, [])):
            return f"«{text}» — у товаров категории «{category}» не бывает такой характеристики"
        return None

    @staticmethod
    def _dimension(text: str, words: list[str]) -> str | None:
        for m in NUMBER_UNIT.finditer(text):
            dim = unit_dimension(m.group(1))
            if dim:
                return dim
        return next((NAME_DIMS[w] for w in words if w in NAME_DIMS), None)
