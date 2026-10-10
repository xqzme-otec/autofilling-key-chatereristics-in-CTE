"""Второй и третий уровень иерархии источников характеристик: Портал поставщиков и новые.

  1. КТРУ   — ml/values.py, ml/units.py (origin = "ktru")
  2. Портал — характеристика есть в справочнике характеристик Портала (вкладка 2 выгрузок):
              «цвет синий» → «Цвет: синий», «синий» → «Цвет: синий» (origin = "portal")
  3. Новая  — фрагмент описания не лёг ни на КТРУ, ни на Портал, но выбрасывать его нельзя:
              «противоскользящая рукоятка» → «Противоскользящая рукоятка: Да» (origin = "new")

    .venv/bin/python -m ml.portal build   # data/raw/supplier_portal/*.xlsx → data/processed/portal_characteristics.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ml.text import tokenize

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "supplier_portal"
DICT_PATH = ROOT / "data" / "processed" / "portal_characteristics.json"
CATEGORIES = {"Инструменты": "инструменты", "Мебель": "мебель", "Одежда": "одежда"}
MIN_USAGE = 2          # характеристика должна встречаться хотя бы в 2 категориях Портала (КПГЗ)
MAX_VALUES = 60        # значений на характеристику храним не больше
SKIP_NAMES = {"артикул", "модель", "производитель", "страна происхождения", "бренд", "торговая марка",
              "гарантийный срок", "срок гарантии", "описание", "комплектация"}


def _lemmas(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalnum()]


def build() -> None:
    """Собирает словарь характеристик Портала по категориям из вкладки 2 выгрузок."""
    import pandas as pd

    out: dict[str, dict] = {}
    for file, category in CATEGORIES.items():
        df = pd.read_excel(RAW_DIR / f"{file}.xlsx", sheet_name=1, dtype=str)
        df = df[df["Тип (базовая/эталонная/категореобразующая)"].isin(["Базовая", "Эталонная"])]
        usage = df.groupby("Название характеристики")["Код КПГЗ"].nunique()
        values: dict[str, Counter] = defaultdict(Counter)
        units: dict[str, Counter] = defaultdict(Counter)
        kinds: dict[str, Counter] = defaultdict(Counter)
        for r in df.itertuples(index=False):
            name = str(r[3]).strip()
            if usage.get(name, 0) < MIN_USAGE or name.lower() in SKIP_NAMES or len(name) > 60:
                continue
            kinds[name][str(r[5])] += 1
            if isinstance(r[7], str) and r[7].strip():
                units[name][r[7].strip()] += 1
            if isinstance(r[6], str) and r[6].strip() and str(r[5]) in ("Текст", "Справочник", "Признак"):
                for v in re.split(r";\s*", r[6].strip()):
                    if 0 < len(v) <= 60:
                        values[name][v.strip()] += 1
        out[category] = {
            name: {"usage": int(usage[name]),
                   "kind": kinds[name].most_common(1)[0][0],
                   # единица — только у числовых: у текстовых в выгрузке бывает мусор («Материал корпуса» — «чугун»)
                   "unit": units[name].most_common(1)[0][0]
                           if units[name] and "числов" in kinds[name].most_common(1)[0][0].lower() else None,
                   "values": [v for v, _ in values[name].most_common(MAX_VALUES)]}
            for name in kinds}
        print(f"{file}: {len(out[category])} характеристик")
    DICT_PATH.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"→ {DICT_PATH} ({DICT_PATH.stat().st_size // 1024} КБ)")


class PortalMatcher:
    def __init__(self, path: Path = DICT_PATH):
        self.by_category: dict[str, list[tuple[list[str], str, dict]]] = {}
        self.value_index: dict[str, dict[str, set]] = {}
        if not path.exists():
            return  # словаря нет — уровень «Портал» пропускается, остаётся «новая»
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, chars in data.items():
            # длинные названия проверяем первыми: «Цвет сиденья» раньше, чем «Цвет»
            names = sorted(((_lemmas(n), n, info) for n, info in chars.items()), key=lambda x: -len(x[0]))
            self.by_category[category] = [x for x in names if x[0]]
            self.usage = getattr(self, "usage", {})
            self.usage[category] = {n: info["usage"] for n, info in chars.items()}
            index: dict[str, set] = defaultdict(set)
            for name, info in chars.items():
                if info["kind"] in ("Текст", "Справочник"):
                    for v in info["values"]:
                        index[" ".join(_lemmas(v))].add(name)
            self.value_index[category] = index

    @property
    def available(self) -> bool:
        return bool(self.by_category)

    def match(self, segment: str, category: str | None) -> dict | None:
        """Фрагмент описания → характеристика Портала {name, value, unit} или None."""
        if not category or category not in self.by_category:
            return None
        toks = [t for t in tokenize(segment) if t["text"][0].isalnum()]
        lemmas = [t["lemma"] for t in toks]
        # 1. фрагмент начинается с названия характеристики: «цвет сиденья синий», «вес 0.6 кг»
        for name_lemmas, name, info in self.by_category[category]:
            n = len(name_lemmas)
            if lemmas[:n] == name_lemmas and len(lemmas) > n:
                value = segment[toks[n]["span"][0]:].strip()
                return {"name": name, "value": value, "unit": info["unit"] if not _has_unit(value) else None}
        # 2. фрагмент — известное значение характеристики: «синий» → «Цвет» (короткое и употребимое название)
        names = self.value_index[category].get(" ".join(lemmas), set())
        if names:
            name = min(names, key=lambda n: (len(n.split()), -self.usage[category].get(n, 0)))
            return {"name": name, "value": segment.strip(), "unit": None}
        return None


def _has_unit(value: str) -> bool:
    return bool(re.search(r"\d\s*[a-zа-я]", value, re.IGNORECASE))


def new_characteristic(segment: str) -> dict:
    """Фрагмент без места в КТРУ и на Портале — сохраняем как новую характеристику, не выбрасываем."""
    text = segment.strip()
    m = re.match(r"^(.*?)[\s:]+(\d[\d.,]*\s*\S*)$", text)  # «длина кабеля 3 м» → имя + числовое значение
    if m and m.group(1):
        return {"name": m.group(1)[:1].upper() + m.group(1)[1:], "value": m.group(2), "unit": None}
    return {"name": text[:1].upper() + text[1:], "value": "Да", "unit": None}


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
