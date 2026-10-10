"""The second and third levels of the characteristic source hierarchy: the supplier portal and new ones.

  1. KTRU   — ml/values.py, ml/units.py (origin = "ktru")
  2. Portal — the characteristic exists in the portal characteristic reference (sheet 2 of the exports):
              «цвет синий» → «Цвет: синий», «синий» → «Цвет: синий» (origin = "portal")
  3. New    — a description fragment fits neither KTRU nor the portal, but must not be dropped:
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
MIN_USAGE = 2          # a characteristic must occur in at least 2 portal categories (KPGZ)
MAX_VALUES = 300       # at most this many values are kept per characteristic
MIN_VALUES_TO_CHECK = 5  # check a value against the portal list only when the list is substantial
SKIP_NAMES = {"артикул", "модель", "производитель", "страна происхождения", "бренд", "торговая марка",
              "гарантийный срок", "срок гарантии", "описание", "комплектация"}


def _lemmas(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalnum()]


def build() -> None:
    """Builds the portal characteristic dictionary by category from sheet 2 of the exports."""
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
                   # a unit only for numeric ones: text ones carry junk in the export («Материал корпуса» — «чугун»)
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
        self.usage: dict[str, dict[str, int]] = {}
        self.value_index: dict[str, dict[str, set]] = {}
        if not path.exists():
            return  # no dictionary — the «portal» level is skipped, «new» remains
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, chars in data.items():
            # long names are checked first: «Цвет сиденья» before «Цвет»
            names = sorted(((_lemmas(n), n, info) for n, info in chars.items()), key=lambda x: -len(x[0]))
            self.by_category[category] = [x for x in names if x[0]]
            self.usage[category] = {n: info["usage"] for n, info in chars.items()}
            index: dict[str, dict] = defaultdict(dict)  # value lemmas → {characteristic: value as on the portal}
            for name, info in chars.items():
                if info["kind"] in ("Текст", "Справочник"):
                    for v in info["values"]:
                        index[" ".join(_lemmas(v))].setdefault(name, v)
            self.value_index[category] = index

    def has(self, category: str | None, name: str) -> bool:
        return bool(category) and name in self.usage.get(category, {})

    @property
    def available(self) -> bool:
        return bool(self.by_category)

    def match(self, segment: str, category: str | None) -> dict | None:
        """Description fragment → a portal characteristic {name, value, unit} or None."""
        if not category or category not in self.by_category:
            return None
        toks = [t for t in tokenize(segment) if t["text"][0].isalnum()]
        lemmas = [t["lemma"] for t in toks]
        # 1. the fragment starts with a characteristic name: «цвет сиденья синий», «вес 0.6 кг»
        for name_lemmas, name, info in self.by_category[category]:
            n = len(name_lemmas)
            if lemmas[:n] == name_lemmas and len(lemmas) > n:
                # «Тип патрона», «Вид материала»: a genitive noun continues a longer name, it is not a value
                if _is_genitive_noun(toks[n]["text"]):
                    continue
                value = segment[toks[n]["span"][0]:].strip()
                out = {"name": name, "value": value, "unit": info["unit"] if not _has_unit(value) else None}
                # a portal reference list exists and the value is far from all of it: «куртка, материал бетон»
                # only against a complete list: a list cut at MAX_VALUES may simply miss a real value
                if info["kind"] in ("Справочник", "Текст") and MIN_VALUES_TO_CHECK <= len(info["values"]) < MAX_VALUES:
                    from rapidfuzz import fuzz, process
                    value_lem = " ".join(lemmas[n:])
                    known = [" ".join(_lemmas(v)) for v in info["values"]]
                    best = process.extractOne(value_lem, known, scorer=fuzz.token_set_ratio)
                    if not best or best[1] < 70:
                        out["unknown_value"] = True
                        out["known_values"] = info["values"][:8]
                return out
        # 2. the fragment is a known value of a characteristic: «синий» → «Цвет» (a short and common name)
        names = self.value_index[category].get(" ".join(lemmas), {})
        if names and not any(ch.isdigit() for ch in segment):
            name = min(names, key=lambda n: (len(n.split()), -self.usage[category].get(n, 0)))
            value = names[name]
            return {"name": name, "value": value[:1].upper() + value[1:], "unit": None}
        return None


def _is_genitive_noun(word: str) -> bool:
    from ml.text import _morph
    p = _morph.parse(word.lower())[0]
    return p.tag.POS == "NOUN" and p.tag.case == "gent"


def _has_unit(value: str) -> bool:
    return bool(re.search(r"\d\s*[a-zа-я]", value, re.IGNORECASE))


def new_characteristic(segment: str) -> dict:
    """A fragment with no place in KTRU or on the portal — kept as a new characteristic, never dropped."""
    text = segment.strip()
    m = re.match(r"^(.*?)[\s:]+(\d[\d.,]*\s*\S*)$", text)  # «длина кабеля 3 м» → name + numeric value
    if m and m.group(1):
        return {"name": m.group(1)[:1].upper() + m.group(1)[1:], "value": m.group(2), "unit": None}
    return {"name": text[:1].upper() + text[1:], "value": "Да", "unit": None}


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
