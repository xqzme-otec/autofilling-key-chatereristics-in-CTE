"""Второй и третий уровень иерархии источников характеристик: Портал поставщиков (СТЕ) и новые.

  1. КТРУ   — ml/values.py, ml/units.py (origin = "ktru")
  2. Портал — характеристика есть в выгрузках СТЕ Портала (origin = "portal"):
              сначала в профиле товара — что реально заполняют в карточках этого товара (вкладка 1):
              «стеклопластик» у молотка → «Материал рукоятки: Стеклопластик», «обрезиненная ручка» →
              «Обрезиненная ручка: Да»; затем в справочнике характеристик категории (вкладка 2):
              «цвет синий» → «Цвет: синий», «синий» → «Цвет: синий»
  3. Новая  — фрагмент описания не лёг ни на КТРУ, ни на Портал, но выбрасывать его нельзя:
              «противоскользящая рукоятка» → «Противоскользящая рукоятка: Да» (origin = "new")

В словарь попадают только актуальные характеристики Портала — правило в is_actual().

    .venv/bin/python -m ml.portal build   # data/raw/supplier_portal/*.xlsx → data/processed/portal_characteristics.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ml.text import pos, tokenize

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "supplier_portal"
DICT_PATH = ROOT / "data" / "processed" / "portal_characteristics.json"
CATEGORIES = {"Инструменты": "инструменты", "Мебель": "мебель", "Одежда": "одежда"}
MIN_USAGE = 2          # в общий словарь категории: характеристика хотя бы в 2 категориях Портала (КПГЗ)
MAX_VALUES = 300       # значений на характеристику храним не больше
MIN_VALUES_TO_CHECK = 5  # check a value against the portal list only when the list is substantial
SKIP_NAMES = {"артикул", "модель", "производитель", "страна происхождения", "бренд", "торговая марка",
              "гарантийный срок", "срок гарантии", "описание", "комплектация"}
CLASS_COLUMN = "Тип (базовая/эталонная/категореобразующая)"
ACTUAL_CLASSES = {"Базовая", "Эталонная"}
SERVICE_CLASS = "Категореобразующая"  # «Вид продукции: Товары» — set by the portal itself, not a product feature
TEXT_KINDS = ("Текст", "Справочник", "Признак")
MAX_NAME_WORDS = 16     # no reference name worth matching is longer
# product profile: what the cards of one product (category + main noun of the name) really fill in
MIN_PRODUCT_CARDS = 20  # a product needs at least this many cards to get a profile
MIN_CHAR_CARDS = 3      # a characteristic is in the profile when at least this many cards of the product have it
MIN_CHAR_SHARE = 0.01   # ...and at least this share of them
PROFILE_VALUES = 8      # typical values kept per characteristic of a product
MIN_VALUE_CARDS = 2     # a value points at a characteristic when this many cards of the product have it there
VALUE_LEAD = 1.5        # ...and it is this many times more frequent there than in the runner-up characteristic
YES_VALUES = {"да", "наличие", "есть", "имеется"}
GENERIC_VALUES = YES_VALUES | {"нет", "отсутствует", "без"}
JUNK_VALUE = re.compile(r"^(\(не задано\)|0)(\s|$)")
TRAILING_ZEROS = re.compile(r"(\d+)\.(\d*?)0+(?=\s|$)")


def _lemmas(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalnum()]


def is_actual(classes: set[str], filled: int) -> bool:
    """Only characteristics that are still in use go into the dictionary. The exports have no «actual» flag
    (KTRU has one, see ml/ktru.py), so the rule is: the portal reference lists the characteristic as
    «Базовая» or «Эталонная», it is not a service one and product cards really fill it in."""
    return bool(classes & ACTUAL_CLASSES) and SERVICE_CLASS not in classes and filled > 0


def clean_value(value: str) -> str:
    """«2.03000 кг» → «2.03 кг», «100.00000 мм» → «100 мм»; an empty, «(не задано)» or zero value → ""."""
    value = TRAILING_ZEROS.sub(lambda m: m.group(1) + ("." + m.group(2) if m.group(2) else ""), value.strip())
    return "" if not value or JUNK_VALUE.match(value) else value


def card_index(names) -> dict[str, str]:
    """Reference names in lower case → as written in the reference."""
    return {name.lower(): name for name in names}


def parse_card(raw: str, index: dict[str, str]) -> list[tuple[str, str]]:
    """Card characteristics «Вес 2.03000 кг | Тип питания от сети» → [(«Вес», «2.03 кг»), («Тип питания»,
    «от сети»)]. The name and the value are not delimited in the export: the name is the longest reference
    name the record starts with («Тип патрона» before «Тип»). Records with an unknown name or without a
    value are skipped."""
    out, seen = [], set()
    for part in raw.split("|"):
        part = part.strip()
        words = part.lower().split(" ")
        name = next((index[prefix] for k in range(min(len(words), MAX_NAME_WORDS), 0, -1)
                     if (prefix := " ".join(words[:k])) in index), None)
        if name is None or name in seen:
            continue
        value = clean_value(part[len(name):])
        if value:
            seen.add(name)
            out.append((name, value))
    return out


def _merge_case(values: Counter | None) -> list[tuple[str, int]]:
    """«Мужской»×5, «мужской»×3 → [(«Мужской», 8)]: one value, the most common spelling."""
    merged: dict[str, Counter] = defaultdict(Counter)
    for value, n in (values or {}).items():
        merged[value.lower()][value] += n
    return sorted(((c.most_common(1)[0][0], sum(c.values())) for c in merged.values()), key=lambda x: -x[1])


def _value_index(values_of: dict[str, list[tuple[str, int]]]) -> dict[str, list[str]]:
    """Value lemmas → [characteristic, value as on the portal] within one product: «стеклопластик» →
    «Материал рукоятки». A value shared by several characteristics almost equally is left out."""
    options: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for name, values in values_of.items():
        for value, n in values:
            key = " ".join(_lemmas(value))
            if n >= MIN_VALUE_CARDS and key and key not in GENERIC_VALUES and len(key.split()) <= 4 \
                    and not any(ch.isdigit() for ch in value):
                options[key].append((n, name, value))
    index = {}
    for key, found in options.items():
        found.sort(reverse=True)
        if len(found) == 1 or found[0][0] >= VALUE_LEAD * found[1][0]:
            index[key] = [found[0][1], found[0][2]]
    return index


def build(holdout: int | None = None, out: Path = DICT_PATH) -> None:
    """Собирает словарь характеристик Портала: справочник по категориям (вкладка 2) и профили товаров по
    реальным карточкам (вкладка 1). holdout=N — карточки с ID, кратным N, не используются (для проверки)."""
    from functools import lru_cache

    import pandas as pd

    from ml.text import main_noun

    product_of = lru_cache(maxsize=None)(main_noun)  # the same function gives product_intent in the pipeline
    result: dict[str, dict] = {}
    for file, category in CATEGORIES.items():
        path = RAW_DIR / f"{file}.xlsx"
        # the reference of characteristics by portal category (КПГЗ)
        df = pd.read_excel(path, sheet_name=1, dtype=str)
        df["Название характеристики"] = df["Название характеристики"].str.strip()
        classes: dict[str, set[str]] = defaultdict(set)
        for name, cls in zip(df["Название характеристики"], df[CLASS_COLUMN]):
            if isinstance(name, str) and name:
                classes[name].add(cls)
        df = df[df[CLASS_COLUMN].isin(ACTUAL_CLASSES)]
        usage = df.groupby("Название характеристики")["Код КПГЗ"].nunique()
        values: dict[str, Counter] = defaultdict(Counter)
        units: dict[str, Counter] = defaultdict(Counter)
        kinds: dict[str, Counter] = defaultdict(Counter)
        aliases: dict[str, set[str]] = defaultdict(set)
        for r in df.itertuples(index=False):
            name = r[3]
            if not isinstance(name, str) or not name or name.lower() in SKIP_NAMES or len(name) > 60:
                continue
            kinds[name][str(r[5])] += 1
            if isinstance(r[9], str) and r[9].strip() and r[9].strip() != name:
                aliases[name].add(r[9].strip())
            if isinstance(r[7], str) and r[7].strip():
                units[name][r[7].strip()] += 1
            if isinstance(r[6], str) and r[6].strip() and str(r[5]) in TEXT_KINDS:
                for v in re.split(r";\s*", r[6].strip()):
                    if 0 < len(v) <= 60:
                        values[name][v.strip()] += 1
        kind_of = {name: counts.most_common(1)[0][0] for name, counts in kinds.items()}

        # product cards: subtype words of the names and the characteristics really filled in
        cards = pd.read_excel(path, sheet_name=0, usecols=["ID СТЕ", "Наименование СТЕ", "Характеристики"], dtype=str)
        index = card_index(classes)
        products: dict[str, Counter] = defaultdict(Counter)
        filled: Counter = Counter()
        cards_of: Counter = Counter()
        in_product: dict[str, Counter] = defaultdict(Counter)
        card_values: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for card_id, title, raw in zip(cards["ID СТЕ"], cards["Наименование СТЕ"], cards["Характеристики"]):
            if not isinstance(title, str):
                continue
            if holdout and str(card_id).isdigit() and int(card_id) % holdout == 0:
                continue
            head = title.split(",")[0][:100]
            words = [t for t in tokenize(head) if t["text"][0].isalpha()][:5]
            noun = next((t["lemma"] for t in words if pos(t["text"]) == "NOUN"), None)
            if noun:
                for t in words:
                    if pos(t["text"]) == "ADJF" and abs(words.index(t) - next(i for i, w in enumerate(words) if w["lemma"] == noun)) <= 2:
                        products[noun][t["lemma"]] += 1
            pairs = parse_card(raw, index) if isinstance(raw, str) else []
            for name, _ in pairs:
                filled[name] += 1
            product = product_of(head)
            if not product:
                continue
            cards_of[product] += 1
            for name, value in pairs:
                in_product[product][name] += 1
                if kind_of.get(name) in TEXT_KINDS and len(value) <= 60:
                    card_values[(product, name)][value] += 1

        profiles = {}
        for product, n_cards in cards_of.items():
            if n_cards < MIN_PRODUCT_CARDS:
                continue
            chars, text_values = {}, {}
            for name, n in in_product[product].most_common():
                if n < max(MIN_CHAR_CARDS, n_cards * MIN_CHAR_SHARE):
                    break
                if name not in kind_of or not is_actual(classes[name], filled[name]):
                    continue
                merged = _merge_case(card_values.get((product, name)))
                chars[name] = {"n": n}
                if merged:
                    chars[name]["values"] = [v for v, _ in merged[:PROFILE_VALUES]]
                # «Обрезиненная ручка: Да» — naming the feature is enough to say it is there
                if kind_of[name] == "Признак" or (merged and merged[0][0].lower() in YES_VALUES):
                    chars[name]["flag"] = 1
                elif merged:
                    text_values[name] = merged
            if chars:
                profiles[product] = {"cards": n_cards, "chars": chars, "by_value": _value_index(text_values)}

        in_profiles = {name for p in profiles.values() for name in p["chars"]}
        out_chars: dict[str, dict] = {}
        for name, kind in kind_of.items():
            # the category-wide list stays narrow: a characteristic of one product must not land on another
            wide = usage.get(name, 0) >= MIN_USAGE
            if not is_actual(classes[name], filled[name]) or not (wide or name in in_profiles):
                continue
            out_chars[name] = {
                "usage": int(usage.get(name, 0)),
                "filled": filled[name],
                "cls": "эталонная" if "Эталонная" in classes[name] else "базовая",
                "kind": kind,
                # единица — только у числовых: у текстовых в выгрузке бывает мусор («Материал корпуса» — «чугун»)
                "unit": units[name].most_common(1)[0][0] if units[name] and "числов" in kind.lower() else None,
                "values": [v for v, _ in values[name].most_common(MAX_VALUES)],
                "aliases": sorted(aliases[name])[:20]}
            if not wide:  # known from the cards of particular products: used only through their profiles
                out_chars[name]["local"] = 1
        print(f"{file}: в справочнике {len(classes)} названий, актуальных в словаре {len(out_chars)} "
              f"(из них только в профилях {sum(1 for c in out_chars.values() if c.get('local'))}), "
              f"товаров с профилем {len(profiles)}")
        out_chars["_products"] = {noun: dict(counts.most_common(100)) for noun, counts in products.items()
                                  if sum(counts.values()) >= 2}
        out_chars["_profiles"] = profiles
        result[category] = out_chars
    out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"→ {out} ({out.stat().st_size // 1024} КБ)")


class PortalMatcher:
    def __init__(self, path: Path = DICT_PATH):
        self.by_category: dict[str, list[tuple[list[str], str, dict]]] = {}
        self.usage: dict[str, dict[str, int]] = {}
        self.value_index: dict[str, dict[str, set]] = {}
        self.products: dict[str, dict[str, dict[str, int]]] = {}
        self.profiles: dict[str, dict[str, dict]] = {}  # category → product → {chars, names, by_value, flags}
        if not path.exists():
            return  # no dictionary — the «portal» level is skipped, «new» remains
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, chars in data.items():
            self.products[category] = chars.pop("_products", {})
            profiles = chars.pop("_profiles", {})
            wide = {n: info for n, info in chars.items() if not info.get("local")}
            # длинные названия проверяем первыми: «Цвет сиденья» раньше, чем «Цвет»
            names = sorted(((_lemmas(alias), n, info) for n, info in wide.items()
                            for alias in [n, *info.get("aliases", [])]), key=lambda x: -len(x[0]))
            self.by_category[category] = [x for x in names if x[0]]
            self.usage[category] = {n: info["usage"] for n, info in wide.items()}
            index: dict[str, dict] = defaultdict(dict)  # value lemmas → {characteristic: value as on the portal}
            for name, info in wide.items():
                if info["kind"] in ("Текст", "Справочник"):
                    for v in info["values"]:
                        index[" ".join(_lemmas(v))].setdefault(name, v)
            self.value_index[category] = index
            name_lemmas = {n: _lemmas(n) for p in profiles.values() for n in p["chars"] if n in chars}
            self.profiles[category] = {}
            for product, p in profiles.items():
                own = sorted(((name_lemmas[n], n, chars[n]) for n in p["chars"] if name_lemmas.get(n)),
                             key=lambda x: -len(x[0]))
                self.profiles[category][product] = {
                    "chars": p["chars"], "names": own, "by_value": p.get("by_value", {}),
                    "flags": {" ".join(sorted(lem)): n for lem, n, _ in own if p["chars"][n].get("flag")}}

    def _profile(self, category: str | None, product: str | None) -> dict | None:
        return self.profiles.get(category, {}).get(product) if category and product else None

    def has(self, category: str | None, name: str, product: str | None = None) -> bool:
        """The portal has such a characteristic: for this product (its cards fill it in) or in the category."""
        profile = self._profile(category, product)
        return bool(profile and name in profile["chars"]) or (bool(category) and name in self.usage.get(category, {}))

    def is_subtype(self, category: str | None, product: str | None, word: str) -> bool:
        return bool(category and product and self.products.get(category, {}).get(product, {}).get(word))

    @property
    def available(self) -> bool:
        return bool(self.by_category)

    def match(self, segment: str, category: str | None, product: str | None = None) -> dict | None:
        """Фрагмент описания → характеристика Портала {name, value, unit} или None.
        product (главное слово товара) — сначала смотрим, что заполняют в карточках этого товара."""
        if not category or category not in self.by_category:
            return None
        toks = [t for t in tokenize(segment) if t["text"][0].isalnum()]
        lemmas = [t["lemma"] for t in toks]
        profile = self._profile(category, product)
        # 1. фрагмент начинается с названия характеристики: «цвет сиденья синий», «вес 0.6 кг»
        found = (profile and self._by_name(profile["names"], segment, toks, lemmas)) \
            or self._by_name(self.by_category[category], segment, toks, lemmas)
        if found:
            return found
        if not lemmas or any(ch.isdigit() for ch in segment):
            return None
        if profile:
            # 2. a value the cards of this product give to one characteristic: «стеклопластик» → «Материал рукоятки»
            hit = profile["by_value"].get(" ".join(lemmas))
            if hit:
                return {"name": hit[0], "value": hit[1][:1].upper() + hit[1][1:], "unit": None}
            # 3. a yes/no characteristic of this product named as is: «обрезиненная ручка» → «Да»
            name = profile["flags"].get(" ".join(sorted(lemmas)))
            if name:
                return {"name": name, "value": "Да", "unit": None}
        # 4. фрагмент — известное значение характеристики: «синий» → «Цвет» (короткое и употребимое название)
        names = self.value_index[category].get(" ".join(lemmas), {})
        if names:
            name = min(names, key=lambda n: (len(n.split()), -self.usage[category].get(n, 0)))
            value = names[name]
            return {"name": name, "value": value[:1].upper() + value[1:], "unit": None}
        return None

    @staticmethod
    def _by_name(entries: list[tuple[list[str], str, dict]], segment: str, toks: list[dict],
                 lemmas: list[str]) -> dict | None:
        for name_lemmas, name, info in entries:
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
    if sys.argv[1:2] == ["build"]:
        import argparse

        parser = argparse.ArgumentParser(prog="python -m ml.portal build")
        parser.add_argument("--holdout", type=int, default=None, help="skip cards whose ID is a multiple of N")
        parser.add_argument("--out", type=Path, default=DICT_PATH)
        args = parser.parse_args(sys.argv[2:])
        build(args.holdout, args.out)
    else:
        print(__doc__)
