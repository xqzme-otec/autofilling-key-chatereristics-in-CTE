from __future__ import annotations

"""Numbers and units → numeric KTRU characteristics.

«высота 110 см» → «Высота: 1000–1200 Миллиметр» (110 cm = 1100 mm falls into the range).

1. Find numbers: «110 см», «0,5кг», «52-54», «от 100 до 200 мм», «3 полки».
2. Pick the characteristic:
      - by the words next to the number («высота 110», «массой 0.5 кг», «3 полки»);
      - without words — by the unit, when the candidates have exactly one characteristic of that
          dimension («молоток 0.5 кг» → «Масса молотка»).
3. Convert to the KTRU unit and find the range the number falls into.
Errors: unit does not fit («длина 2 кг»), number outside all allowed ranges.
Numbers that have no place in KTRU for the found products («стол письменный высотой 75 см» — the position
has no height characteristic) are returned separately: the source hierarchy keeps them (portal / new).
"""
import os
import re
from itertools import permutations

from ml.ktru import KtruStore, format_value, is_text_char
from ml.numwords import find_number_words
from ml.text import main_noun, pos, tokenize

# KTRU unit (OKEI) → (dimension, factor to the base unit of the dimension)
KTRU_UNITS = {
    "Миллиметр": ("length", 1), "Сантиметр": ("length", 10), "Метр": ("length", 1000),
    "Километр; тысяча метров": ("length", 1_000_000), "Дюйм (25,4 мм)": ("length", 25.4),
    "Миллиграмм": ("mass", 0.000001), "Грамм": ("mass", 0.001), "Килограмм": ("mass", 1), "Тонна": ("mass", 1000),
    "Ватт": ("power", 1), "Киловатт": ("power", 1000), "Вольт": ("voltage", 1),
    "Ампер-час (3,6 кКл)": ("charge", 1), "Оборот в минуту": ("rpm", 1), "Штука": ("count", 1),
    "Градус Цельсия": ("temp", 1), "Квадратный миллиметр": ("area", 1), "Минута": ("time", 1),
}

# how users write units → (dimension, factor)
USER_UNITS = {
    "мм": ("length", 1), "миллиметр": ("length", 1), "см": ("length", 10), "сантиметр": ("length", 10),
    "дм": ("length", 100), "м": ("length", 1000), "метр": ("length", 1000), "км": ("length", 1_000_000),
    '"': ("length", 25.4), "дюйм": ("length", 25.4),
    "мкг": ("mass", 1e-9), "мг": ("mass", 1e-6), "миллиграмм": ("mass", 1e-6), "микрограмм": ("mass", 1e-9),
    "г": ("mass", 0.001), "гр": ("mass", 0.001), "грамм": ("mass", 0.001), "кг": ("mass", 1),
    "килограмм": ("mass", 1), "т": ("mass", 1000), "тонна": ("mass", 1000),
    "вт": ("power", 1), "ватт": ("power", 1), "квт": ("power", 1000), "киловатт": ("power", 1000),
    "в": ("voltage", 1), "вольт": ("voltage", 1),
    "ач": ("charge", 1), "а*ч": ("charge", 1), "мач": ("charge", 0.001),
    "об/мин": ("rpm", 1), "шт": ("count", 1), "штука": ("count", 1),
    "°c": ("temp", 1), "°с": ("temp", 1), "градус": ("temp", 1), "мм2": ("area", 1),
}
# full words come with endings («сантиметров», «килограмма»), abbreviations without
UNIT_WORDS = ["миллиметр", "сантиметр", "метр", "дюйм", "миллиграмм", "микрограмм", "килограмм", "грамм", "тонн",
              "киловатт", "ватт",
              "вольт", "штук", "градус"]
USER_UNITS.update({"тонн": ("mass", 1000), "штук": ("count", 1)})
UNIT_ABBR = sorted((u for u in USER_UNITS if u not in UNIT_WORDS and u not in ("тонна", "штука")), key=len, reverse=True)
NUM = r"\d+(?:[.,]\d+)?"
NUMBER_RE = re.compile(
    rf"(?:(?P<from>от)\s+)?(?P<a>{NUM})(?:\s*(?:-|до|по)\s*(?P<b>{NUM}))?"
    rf"(?:\s*(?P<unit>(?:{'|'.join(UNIT_WORDS)})[а-я]*|(?:{'|'.join(map(re.escape, UNIT_ABBR))})(?![a-zа-я0-9])))?",
    re.IGNORECASE)
GENERIC = {"наличие", "тип", "вид", "количество", "число", "максимальный", "минимальный", "общий",
           "без", "учет", "предельный", "отклонение", "для", "с", "и", "в", "на", "из", "до", "по", "мм", "см"}
CONTEXT_BEFORE, CONTEXT_AFTER = 3, 2
# a number with a unit but no keyword is assigned only to a characteristic whose name means that dimension:
# «стол 1.5 кг» must not land in «Максимальная нагрузка на столешницу»
DIMENSION_WORDS = {"mass": ("масса", "вес"), "power": ("мощность",), "voltage": ("напряжение",),
                   "charge": ("емкость",), "rpm": ("частота", "оборот", "скорость")}
DIMENSION_NAMES = {"mass": "Масса", "power": "Мощность", "voltage": "Напряжение", "charge": "Емкость аккумулятора",
                   "rpm": "Частота вращения", "length": "Размер", "temp": "Температура", "count": "Количество"}
# a value in an open KTRU range («≥ 100 Сантиметр») that is this many times beyond the outermost bound
# is considered suspicious: «стол высотой 5 метров», «молоток 50 кг»
SUSPICIOUS_FACTOR = 3


SHORT_UNITS = {("mass", 1): "кг", ("mass", 0.001): "г", ("mass", 1e-6): "мг", ("mass", 1e-9): "мкг", ("mass", 1000): "т", ("length", 1): "мм",
               ("length", 10): "см", ("length", 100): "дм", ("length", 1000): "м", ("length", 25.4): "дюйм",
               ("power", 1): "Вт", ("power", 1000): "кВт", ("voltage", 1): "В", ("charge", 1): "А·ч",
               ("charge", 0.001): "мА·ч", ("rpm", 1): "об/мин", ("count", 1): "шт", ("temp", 1): "°C"}


def _same_noun(noun: str, word: str) -> bool:
    """«шкаф» names «Шкаф для одежды» and «Шкаф-купе», «матрас» names «Матрац» (one letter differs at the end),
    but «электрод» does not name «Дрель электрическая»."""
    if word == noun or word.startswith(noun + "-") or word.endswith("-" + noun):
        return True
    short = min(len(noun), len(word))
    return short >= 5 and len(os.path.commonprefix([noun, word])) >= short - 1


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def contains(v: dict, x: float) -> bool:
    lo, hi = v.get("min"), v.get("max")
    if lo is not None and (x < lo or (x == lo and not v.get("min_incl", True))):
        return False
    if hi is not None and (x > hi or (x == hi and not v.get("max_incl", True))):
        return False
    return True


def contains_range(v: dict, lo: float, hi: float) -> bool:
    """User range [lo, hi] lies inside a KTRU value; the upper end may touch an exclusive bound."""
    if not contains(v, lo):
        return False
    vmax = v.get("max")
    return vmax is None or hi < vmax or hi == vmax


def _inside_code(text: str, start: int, end: int, has_unit: bool) -> bool:
    """The number is part of a model/article code: glued to letters, dashes, slashes («П-30-950», «2930-H5», «HR2450FT»)."""
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return before.isalpha() or before in "-/_" or after in "-/_" or (after.isalpha() and not has_unit)


def _is_decimal_sep(text: str, i: int) -> bool:
    return 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()


def _last_sep(text: str, before: int) -> int:
    """Index right after the last fragment separator before `before` (commas inside numbers do not count)."""
    for i in range(before - 1, -1, -1):
        if text[i] in ",;\n" and not _is_decimal_sep(text, i):
            return i + 1
    return 0


def _next_sep(text: str, after: int) -> int:
    for i in range(after, len(text)):
        if text[i] in ",;\n" and not _is_decimal_sep(text, i):
            return i
    return len(text)


def _mentions(text: str) -> list[dict]:
    """Number mentions: digits («1,5кг», «52-54», «от 100 до 200 мм») and words («полтора килограмма»)."""
    out = []
    for m in NUMBER_RE.finditer(text):
        out.append({"start": m.start(), "end": m.end(), "a": _num(m.group("a")),
                    "b": _num(m.group("b")) if m.group("b") else None, "unit_raw": m.group("unit"),
                    "digits": len(re.split(r"[.,]", m.group("a"))[0])})
    taken = [(x["start"], x["end"]) for x in out]
    for w in find_number_words(text, lambda word: _user_unit(word) is not None and len(word) > 1):
        if not any(s < w["end"] and w["start"] < e for s, e in taken):
            out.append({"start": w["start"], "end": w["end"], "a": w["value"], "b": None,
                        "unit_raw": w["unit_raw"], "digits": 1})
    return sorted(out, key=lambda x: x["start"])


def _user_unit(raw: str | None):
    if not raw:
        return None
    raw = raw.lower().rstrip(".")
    if raw in USER_UNITS:
        return USER_UNITS[raw]
    return next((USER_UNITS[w] for w in UNIT_WORDS if raw.startswith(w)), None)


# ---------- dimensions: «120x60x75 см», «1200х600х750», «Ш120 Г60 В75», «120 на 60» ----------

AXES = ("Длина", "Ширина", "Глубина", "Высота")
AXIS_LABELS = {"д": "Длина", "ш": "Ширина", "г": "Глубина", "в": "Высота"}
DIM_UNIT = r"(?:мм|см|дм|м|mm|cm|миллиметр[а-я]*|сантиметр[а-я]*|метр[а-я]*)(?![a-zа-я0-9])"
DIM_UNITS = {"mm": ("length", 1), "cm": ("length", 10)}
_DIM_SEP = r"\s*[xх*]\s*"  # latin x, cyrillic х, «*»; «×» is normalized to «x» by ml.text.normalize
_DIM_PART = rf"({NUM})(?:\s*({DIM_UNIT}))?"
DIMS_RE = re.compile(
    rf"(?<![\w.,/-]){_DIM_PART}(?:{_DIM_SEP}{_DIM_PART}(?:{_DIM_SEP}{_DIM_PART})?|\s+на\s+{_DIM_PART})"
    rf"(?![\w/-]|[.,]\d)", re.IGNORECASE)
# a label is not glued to a preceding letter, except the «х» separator: «Ш120хГ60хВ75»
LABELED_RE = re.compile(rf"(?:^|(?<=[^A-Za-zА-Яа-яЁё])|(?<=[xхXХ]))([ДШГВдшгв])\.?\s*[:=]?\s*{_DIM_PART}"
                        rf"(?![^\W_xхXХ]|[/-]|[.,]\d)", re.IGNORECASE)
HEADER_RE = re.compile(r"(?<![^\W\d_])([ДШГВдшгв])\s*[xх*]\s*([ДШГВдшгв])(?:\s*[xх*]\s*([ДШГВдшгв]))?(?![^\W\d_])")
# without a unit: numbers above this are millimetres («1200х600х750»), otherwise centimetres («120х60х75»)
MM_THRESHOLD = 300
# conventional order of overall dimensions — used on ties: Д/Ш × Г × В («120х60х75» = длина × ширина × высота)
ORDER_PREF = {
    3: ({"Длина": 0, "Ширина": 1, "Высота": 2, "Глубина": 3},
        {"Ширина": 0, "Глубина": 0, "Длина": 2, "Высота": 3},
        {"Высота": 0, "Глубина": 1, "Ширина": 2, "Длина": 3}),
    2: ({"Длина": 0, "Ширина": 1, "Высота": 2, "Глубина": 2},
        {"Ширина": 0, "Глубина": 0, "Высота": 1, "Длина": 2}),
}
OVERALL_SIZE_NAME = "Габаритные размеры"  # how the supplier portal calls the whole «Д×Ш×В» value
# words that still mean the whole product: «Длина столешницы», «Высота до столешницы»,
# «Ширина спального места» (the product name itself — «Высота шкафа» — is not counted)
OVERALL_QUALIFIERS = {"столешница", "спальный", "место"}
# words naming a dimension group before it («габариты», «размер спального места») and packaging sizes
DIM_NAME_WORDS = {"габарит", "габаритный", "размер"}
PACKAGING_WORDS = {"упаковка", "коробка", "тара"}
# words that may stand before the overall dimensions in a fragment: «Габаритные размеры изделия (ДхШхВ), мм»
PREFIX_WORDS = DIM_NAME_WORDS | OVERALL_QUALIFIERS | {"изделие", "товар", "общий", "внешний", "наружный", "в", "сбор",
                                                      "собранный", "вид", "мм", "см", "м"}
AXES_HEADER_RE = re.compile(r"[ДШГВдшгв](?:[xхXХ][ДШГВдшгв]){1,2}")


def _dim_unit(raw: str | None):
    if not raw:
        return None
    return DIM_UNITS.get(raw.lower()) or _user_unit(raw)


def find_dimensions(text: str) -> list[dict]:
    """Dimension groups: [{start, end, nums: [float], units: [unit|None], labels: [axis|None]}].

    A unit written after any number applies to the numbers without their own unit.
    """
    groups = []
    for m in LABELED_RE.finditer(text):  # «Ш120 Г60 В75», «д190*ш71*в46»
        part = {"start": m.start(), "end": m.end(), "num": _num(m.group(2)), "unit": m.group(3),
                "label": AXIS_LABELS[m.group(1).lower()]}
        last = groups[-1] if groups else None
        if last and re.fullmatch(r"[\s,;xх*/]*", text[last["end"]:m.start()], re.IGNORECASE) \
                and part["label"] not in last["labels"] and len(last["nums"]) < 3:
            last["end"] = m.end()
            last["nums"].append(part["num"])
            last["units"].append(part["unit"])
            last["labels"].append(part["label"])
        else:
            groups.append({"start": m.start(), "end": m.end(), "nums": [part["num"]], "units": [part["unit"]],
                           "labels": [part["label"]]})
    groups = [g for g in groups if len(g["nums"]) >= 2]  # a single «В 220» is not a dimension
    for m in DIMS_RE.finditer(text):
        if any(g["start"] < m.end() and m.start() < g["end"] for g in groups):
            continue
        raw = [(m.group(i), m.group(i + 1)) for i in (1, 3, 5, 7) if m.group(i)]
        # «2x20 Вт», «3х2.5 мм2» — not a size: a unit of another kind right after the group
        after = re.match(r"\s*([^\s,;]+)", text[m.end():])
        if after and _user_unit(after.group(1)) and _user_unit(after.group(1))[0] != "length":
            continue
        g = {"start": m.start(), "end": m.end(), "nums": [_num(a) for a, _ in raw], "units": [u for _, u in raw],
             "labels": [None] * len(raw)}
        # «ШхГхВ 500х600х1800», «габариты (ДхШхВ), мм: 1200х600х750» — axis order written before the numbers
        offset = max(0, m.start() - 30)
        h = list(HEADER_RE.finditer(text[offset:m.start()]))
        tail = text[offset + h[-1].end():m.start()] if h else ""
        if h and re.fullmatch(r"[\s:()=,.\-a-zа-я]*", tail, re.IGNORECASE):
            letters = [x for x in h[-1].groups() if x]
            if len(letters) == len(raw):
                g["labels"] = [AXIS_LABELS[x.lower()] for x in letters]
                g["start"] = offset + h[-1].start()
                unit = re.search(rf"(?<![а-яa-z]){DIM_UNIT}", tail, re.IGNORECASE)
                if unit and not any(g["units"]):
                    g["units"][-1] = unit.group()
        groups.append(g)
    for g in groups:
        explicit = [u for u in g["units"] if _dim_unit(u)]
        default = _dim_unit(explicit[-1]) if explicit else None
        g["unit_given"] = default is not None
        if default is None:
            default = ("length", 1) if max(g["nums"]) > MM_THRESHOLD else ("length", 10)
        g["units"] = [_dim_unit(u) or default for u in g["units"]]
        if any(u[0] != "length" for u in g["units"]):
            g["units"] = None
    return sorted((g for g in groups if g["units"]), key=lambda g: g["start"])


class NumberMatcher:
    def __init__(self, store: KtruStore):
        self.store = store

    def _num_chars(self, codes: list[str]) -> dict[str, dict]:
        """Numeric characteristics of the candidates: name → {unit, keys, by_code: {code: values}}."""
        chars: dict[str, dict] = {}
        for code in codes:
            item = self.store.by_code[code]
            name_words = {t["lemma"] for t in tokenize(item["name"])}
            for ch in item["characteristics"]:
                if is_text_char(ch):
                    continue
                info = chars.setdefault(ch["name"], {"unit": ch["unit"], "keys": None, "by_code": {}, "name_words": set()})
                info["by_code"][code] = (ch["unit"], ch["values"])
                info["name_words"] |= name_words
        for name, info in chars.items():
            lem = [t["lemma"] for t in tokenize(name) if t["text"][0].isalpha()]
            keys = [l for l in lem if l not in GENERIC and l not in info["name_words"]]
            info["keys"] = keys or [l for l in lem if l not in GENERIC]
        return chars

    @staticmethod
    def _key_score(context: list[str], keys: list[str]) -> int:
        def eq(a, b):
            return a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])
        return sum(any(eq(c, k) for c in context) for k in keys)

    @staticmethod
    def _axis_chars(chars: dict[str, dict]) -> dict[str, str]:
        """Overall dimension characteristics of the candidates: axis → name.

        «Высота шкафа» or «Высота (без учета регулировки)» is the overall height; «Высота выдвижного ящика» is a part.
        Per axis the most general one wins (fewest key words besides the axis and the product name).
        """
        best: dict[str, tuple] = {}
        for name, info in chars.items():
            axis = name.split()[0].strip(",").capitalize()
            if axis not in AXES or KTRU_UNITS.get(info["unit"], (None,))[0] != "length":
                continue
            lem = [t["lemma"] for t in tokenize(re.sub(r"\(.*?\)", "", name)) if t["text"][0].isalpha()]
            keys = [l for l in lem if l not in GENERIC and l not in info["name_words"]]
            if not set(keys[1:]) <= OVERALL_QUALIFIERS:
                continue  # «Высота спинки», «Высота шрифта» — a part, not the whole product
            key = (len(keys), -len(info["by_code"]), name)
            if axis not in best or key < best[axis]:
                best[axis] = key
        return {axis: key[2] for axis, key in best.items()}

    @staticmethod
    def _label_char(label: str, axis_chars: dict[str, str]) -> str | None:
        """KTRU characteristic for an explicitly labelled axis: «Ш» is «Длина» for a cabinet that has
        «Длина × Глубина × Высота», «Г» is «Ширина» for a table that has «Длина × Ширина»."""
        if label in axis_chars:
            return axis_chars[label]
        fallback = {"Ширина": "Длина", "Длина": "Ширина", "Глубина": "Ширина" if "Длина" in axis_chars else None}
        alt = fallback.get(label)
        if alt and alt in axis_chars and not (label == "Длина" and "Глубина" in axis_chars):
            return axis_chars[alt]
        return None

    @staticmethod
    def _fit(info: dict, mm: float) -> int:
        """How well a length (in mm) fits a characteristic over the candidates: 2 — inside a bounded KTRU range,
        1 — inside an open range, 0 — suspiciously far in an open range, -2 — outside every range."""
        best = -2
        for ktru_unit, values in info["by_code"].values():
            dim, factor = KTRU_UNITS.get(ktru_unit, (None, 1))
            if dim != "length":
                continue
            x = round(mm / factor, 9)
            hit = next((v for v in values if contains(v, x)), None)
            if hit is None:
                continue
            if hit.get("min") is not None and hit.get("max") is not None:
                return 2
            best = max(best, 0 if _suspicious("", hit, values, x, x, ktru_unit, "", None) else 1)
        return best

    def _assign_axes(self, nums_mm: list[float], labels: list, chars: dict, axis_chars: dict) -> list[tuple]:
        """→ [(axis, KTRU characteristic or None)] for each number of a dimension group."""
        n = len(nums_mm)
        if any(labels):
            return [(label, self._label_char(label, axis_chars) if label else None) for label in labels]
        available = [a for a in AXES if a in axis_chars]
        # axes without a KTRU characteristic still get a conventional name (they go to the portal / new level)
        pad = [a for a in ("Длина", "Ширина", "Высота", "Глубина") if a not in available
               and not (a == "Ширина" and "Глубина" in available) and not (a == "Глубина" and "Ширина" in available)]
        if not available:
            return [(a, None) for a in pad[:n]]
        pref = ORDER_PREF[n]

        def score(perm):
            fit = sum(self._fit(chars[axis_chars[a]], x) if a in axis_chars else 0 for a, x in zip(perm, nums_mm))
            return fit, -sum(pref[i][a] for i, a in enumerate(perm))

        # padding only when the candidates have fewer axes than numbers: otherwise a number outside the KTRU ranges
        # would escape to a padded axis instead of being reported.
        # Three horizontal axes at once is not a reading of «Д×Ш×В» (a mixed pool may have «Длина», «Ширина» and
        # «Глубина»): the height must be one of the three
        extra = max(0, n - len(available))
        while True:
            pool = available + pad[:extra]
            perms = [p for p in permutations(pool, n) if not {"Длина", "Ширина", "Глубина"} <= set(p)]
            if perms or extra >= len(pad):
                break
            extra += 1
        best = max(perms, key=score) if perms else available[:n]
        return [(a, axis_chars.get(a)) for a in best]

    def _match_dimensions(self, text: str, g: dict, chars: dict, rank: dict, category: str | None,
                          tokens: list[dict], first: bool):
        """A dimension group → (characteristics, errors, unmatched, reserved): each number goes to an axis
        characteristic of the candidates; the order «Д×Ш×В» vs «Ш×Г×В» vs «В×Ш×Г» is chosen by how the numbers fit
        the KTRU ranges."""
        found, errors, unmatched = [], [], []
        source = text[g["start"]:g["end"]].strip()
        span = [g["start"], g["start"] + len(source)]
        # words naming the group in its fragment: «Размер спального места 160*80», «Габариты в упаковке 159x113x29»
        frag_start = _last_sep(text, g["start"])
        prefix = [t for t in tokens if frag_start <= t["span"][0] and t["span"][1] <= g["start"] and t["text"][0].isalpha()]
        named_from = next((i for i, t in enumerate(prefix) if t["lemma"] in DIM_NAME_WORDS), None)
        reserved = [t["span"] for t in prefix[named_from:]] if named_from is not None else []
        packaging = next((i for i, t in enumerate(prefix) if t["lemma"] in PACKAGING_WORDS), None)
        # the first group is the product's; a later one outside the product name, after other words, is a part's:
        # «Кровать 1900х800, …, металлический уголок 140мм*120мм»
        part = not first and g["start"] >= _next_sep(text, 0) and any(
            t["lemma"] not in PREFIX_WORDS and not AXES_HEADER_RE.fullmatch(t["text"]) for t in prefix)
        axis_chars = self._axis_chars(chars)
        # furniture and clothing sizes are «Д×Ш×В»; a tool's «14 x 150 мм» may be diameter × length, so a tool gets
        # axes only when its KTRU characteristics cover every number
        unknown_order = category not in ("мебель", "одежда") and len(axis_chars) < len(g["nums"])
        if packaging is not None or part or (not any(g["labels"]) and (unknown_order or (
                not axis_chars and category != "мебель"))):
            # sizes of the packaging or of a part are not the product's; without axis characteristics and axis
            # names — keep the whole value under its own name («Габариты в упаковке») or as the portal calls it
            value = "x".join(_fmt(x) for x in g["nums"]) + (f" {SHORT_UNITS[g['units'][-1]]}" if g["unit_given"] else "")
            keyword, keyword_span = OVERALL_SIZE_NAME, None
            if packaging is not None or part:
                first = 0 if part else min(packaging, named_from) if named_from is not None else packaging
                keyword_span = [prefix[first]["span"][0], prefix[-1]["span"][1]]
                keyword = text[keyword_span[0]:keyword_span[1]]
                reserved = [t["span"] for t in prefix[first:]]
            unmatched.append({"source": value, "span": span, "a": g["nums"][0], "b": None, "unit_raw": None,
                              "value_text": value, "dimension": "length", "number_spans": [], "synthetic": True,
                              "keyword": keyword, "keyword_span": keyword_span})
            return found, errors, unmatched, reserved
        nums_mm = [x * u[1] for x, u in zip(g["nums"], g["units"])]
        for x, unit, (axis, name) in zip(g["nums"], g["units"], self._assign_axes(nums_mm, g["labels"], chars, axis_chars)):
            value_text = f"{_fmt(x)} {SHORT_UNITS[unit]}"
            if name:
                entry, err = self._resolve(name, chars[name], x, None, unit, value_text, span, rank)
                if entry:
                    found.append({**entry, "source": source, "method": "dimensions", "value_text": value_text,
                                  "axis": axis})
                if err:
                    errors.append({**err, "source": source})
            else:
                unmatched.append({"source": value_text, "span": span, "a": x, "b": None, "unit_raw": None,
                                  "value_text": value_text, "dimension": "length", "number_spans": [],
                                  "synthetic": True, "keyword": axis, "keyword_span": None})
        return found, errors, unmatched, reserved

    def match(self, text: str, codes: list[str], skip: list[list[int]] = ()) -> tuple[list[dict], list[dict], list[list[int]], list[dict]]:
        """→ (characteristics, errors, spans of words taken by numbers — text characteristics do not use them,
        numbers without a place in KTRU for these products — for portal / new characteristics).
        skip — spans already taken by another matcher (clothing sizes)."""
        chars = self._num_chars(codes)
        rank = {c: i for i, c in enumerate(codes)}
        product_words = {t["lemma"] for c in codes for t in tokenize(self.store.by_code[c]["name"])}
        found, errors, reserved, unmatched = [], [], [], []
        tokens = tokenize(text)
        category = self.store.by_code[codes[0]]["category"] if codes else None
        taken = [list(s) for s in skip]

        groups = find_dimensions(text)
        if groups:
            # dimensions belong to the product itself: only candidates named by the product noun of the request
            # («сверло 3 x 61 мм» must not land in «Ширина тумбы» of a stray furniture candidate); no such
            # candidate («электрод 2х350мм») — no axis characteristics at all
            noun = main_noun(text.split(",")[0])
            dim_codes = [c for c in codes if any(_same_noun(noun, t["lemma"]) for t in tokenize(self.store.by_code[c]["name"]))] \
                if noun else [c for c in codes if self.store.by_code[c]["name"] == self.store.by_code[codes[0]]["name"]]
            dim_chars = chars if dim_codes == codes else self._num_chars(dim_codes)
        first = True
        for g in groups:
            if any(s < g["end"] and g["start"] < e for s, e in taken):
                continue
            # a bare pair without a unit («10х12») is a size only for furniture; for tools it is a model or a key size
            if len(g["nums"]) == 2 and not g["unit_given"] and not any(g["labels"]) and category != "мебель":
                continue
            f, e, u, r = self._match_dimensions(text, g, dim_chars, rank, category, tokens, first)
            first = False
            found += f
            errors += e
            unmatched += u
            taken.append([g["start"], g["end"]])
            reserved += r + [t["span"] for t in tokens if g["start"] <= t["span"][0] and t["span"][1] <= g["end"]]

        for m in _mentions(text):
            if any(s < m["end"] and m["start"] < e for s, e in taken):
                continue
            a, b, unit, unit_raw = m["a"], m["b"], _user_unit(m["unit_raw"]), m["unit_raw"]
            # words around the number within the fragment (up to a comma that is not a decimal separator)
            seg_start = max(_last_sep(text, m["start"]), 0)
            seg_end = _next_sep(text, m["end"])
            ctx_tokens = ([t for t in tokens if seg_start <= t["span"][0] and t["span"][1] <= m["start"]
                           and t["text"][0].isalpha()][-CONTEXT_BEFORE:]
                          + [t for t in tokens if m["end"] <= t["span"][0] and t["span"][1] <= seg_end
                             and t["text"][0].isalpha() and not _user_unit(t["text"])][:CONTEXT_AFTER])
            context = [t["lemma"] for t in ctx_tokens if t["lemma"] not in product_words]
            source = text[m["start"]:m["end"]].strip()
            span = [m["start"], m["start"] + len(source)]
            if unit is None and (m["digits"] >= 5 or _inside_code(text, m["start"], m["end"], False)):
                continue  # «RUKO 108225», «П-30-950» — an article or a model, not a value
            number_spans = [t["span"] for t in tokens if span[0] <= t["span"][0] and t["span"][1] <= span[1]]
            keyword_tok = next((t for t in reversed(ctx_tokens) if t["span"][1] <= m["start"]
                                and t["lemma"] not in product_words and pos(t["text"]) == "NOUN"), None)

            value_text = _fmt(a) + (f"–{_fmt(b)}" if b is not None else "") + \
                (f" {SHORT_UNITS.get(unit, unit_raw)}" if unit else "")
            unassigned = {"source": source, "span": span, "a": a, "b": b, "unit_raw": unit_raw, "value_text": value_text,
                          "dimension": unit[0] if unit else None, "number_spans": number_spans,
                          "keyword": keyword_tok["text"] if keyword_tok else None,
                          "keyword_span": keyword_tok["span"] if keyword_tok else None}
            scored = [(self._key_score(context, info["keys"]), name, info) for name, info in chars.items()]
            named = [x for x in scored if x[0] > 0]
            if named:
                best_score = max(x[0] for x in named)
                options = [x for x in named if x[0] == best_score]
            elif unit:  # no keyword — by unit, only into a characteristic that means this dimension
                words = DIMENSION_WORDS.get(unit[0], ())
                options = [x for x in scored if KTRU_UNITS.get(x[2]["unit"], (None,))[0] == unit[0]
                           and any(w in x[1].lower() for w in words)]
                # «Масса молотка» vs «Масса головки молотка» — take the most general (fewest key words)
                fewest = min((len(x[2]["keys"]) for x in options), default=0)
                options = [x for x in options if len(x[2]["keys"]) == fewest]
                if len({x[1] for x in options}) != 1:
                    unmatched.append(unassigned)  # «2 м» — length? width? — do not guess
                    continue
            else:
                continue  # bare number without keyword and unit
            if not options:
                unmatched.append(unassigned)
                continue

            if named:
                keys = {k for x in options for k in x[2]["keys"]}
                reserved += [t["span"] for t in ctx_tokens if self._key_score([t["lemma"]], list(keys))]
            # among the keyword matches — the ones compatible by unit
            compatible = [x for x in options if unit is None or KTRU_UNITS.get(x[2]["unit"], (None,))[0] == unit[0]]
            if not compatible:
                name, info = options[0][1], options[0][2]
                reserved += number_spans
                errors.append({"type": "unit_mismatch", "characteristic": name, "source": source, "span": span,
                               "message": f"Единица «{unit_raw}» не подходит для характеристики «{name}» "
                                          f"(ожидается {info['unit'] or 'безразмерное число'})."})
                continue
            # the most general characteristic first, then the best-ranked candidate
            name = min(compatible, key=lambda x: (len(x[2]["keys"]), min(rank[c] for c in x[2]["by_code"])))[1]
            entry, err = self._resolve(name, chars[name], a, b, unit, source, span, rank)
            reserved += number_spans
            if entry:
                found.append(entry)
            if err:
                errors.append(err)
        # one characteristic — one value (the first mention)
        uniq = {}
        for f in found:
            uniq.setdefault(f["name"], f)
        return list(uniq.values()), errors, reserved, unmatched

    @staticmethod
    def _resolve(name, info, a, b, unit, source, span, rank):
        codes, best, warning = [], None, None
        for code, (ktru_unit, values) in sorted(info["by_code"].items(), key=lambda kv: rank[kv[0]]):
            dim, factor = KTRU_UNITS.get(ktru_unit, (None, 1))
            conv = (lambda x: round(x * unit[1] / factor, 9)) if unit and dim == unit[0] else (lambda x: x)
            xa = conv(a)
            xb = conv(b) if b is not None else None
            # for a range «52-54» the value must cover both bounds
            hit = next((v for v in values if (contains(v, xa) if xb is None else contains_range(v, xa, xb))), None)
            if hit:
                codes.append(code)
                if best is None:
                    user_val = _fmt(xa) + (f"–{_fmt(xb)}" if xb is not None else "")
                    best = (format_value(hit), ktru_unit, user_val)
                    warning = _suspicious(name, hit, values, xb if xb is not None else xa, xa, ktru_unit, source, span)
        if best is None:
            ktru_unit, values = next(iter(info["by_code"].values()))
            allowed = ", ".join(format_value(v, ktru_unit) for v in values[:12]) + (" …" if len(values) > 12 else "")
            return None, {"type": "out_of_range", "characteristic": name, "source": source, "span": span,
                          "message": f"Значение «{source}» не входит в допустимые значения характеристики «{name}». "
                                     f"Допустимые: {allowed}."}
        value, ktru_unit, user_val = best
        return {"name": name, "value": value, "unit": ktru_unit, "user_value": f"{user_val} {ktru_unit or ''}".strip(),
                "source": source, "span": span, "method": "number", "score": 95, "codes": codes}, warning


def _suspicious(name, hit, values, hi_x, lo_x, unit, source, span) -> dict | None:
    """The value formally fits an open KTRU range but is too far from its bound."""
    bounds = [b for v in values for b in (v.get("min"), v.get("max")) if b is not None]
    if not bounds:
        return None
    top, bottom = max(bounds), min(b for b in bounds if b > 0) if any(b > 0 for b in bounds) else None
    u = f" {unit}" if unit else ""
    if hit.get("max") is None and top > 0 and hi_x >= SUSPICIOUS_FACTOR * top:
        detail = f"это в {_fmt(hi_x / top)} раз больше самой крупной границы КТРУ ({_fmt(top)}{u})"
    elif hit.get("min") is None and bottom and 0 < lo_x <= bottom / SUSPICIOUS_FACTOR:
        detail = f"это в {_fmt(bottom / lo_x)} раз меньше самой мелкой границы КТРУ ({_fmt(bottom)}{u})"
    else:
        return None
    return {"type": "suspicious_value", "severity": "warning", "characteristic": name, "source": source, "span": span,
            "message": f"Значение «{source}» для характеристики «{name}» выглядит подозрительно: {detail}. "
                       f"Проверьте число и единицу измерения."}
