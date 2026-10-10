from __future__ import annotations

"""Числа и единицы измерения → числовые характеристики КТРУ.

«высота 110 см» → «Высота: 1000–1200 Миллиметр» (110 см = 1100 мм попадает в диапазон).

1. Находим числа: «110 см», «0,5кг», «52-54», «от 100 до 200 мм», «3 полки».
2. Определяем характеристику:
   - по словам рядом с числом («высота 110», «массой 0.5 кг», «3 полки»);
   - если слов нет — по единице, когда у кандидатов ровно одна характеристика такой
     размерности («молоток 0.5 кг» → «Масса молотка»).
3. Переводим в единицу КТРУ и ищем диапазон, в который попадает число.
Ошибки: единица не подходит («длина 2 кг»), число вне всех допустимых диапазонов.
Числа, которым нет места в КТРУ у найденных товаров («стол письменный высотой 75 см» — у позиции
нет характеристики высоты), возвращаются отдельно: их сохраняет иерархия источников (Портал / новая).
"""
import re

from ml.ktru import KtruStore, format_value, is_text_char
from ml.numwords import find_number_words
from ml.text import pos, tokenize

# единица КТРУ (ОКЕИ) → (размерность, множитель к базовой единице размерности)
KTRU_UNITS = {
    "Миллиметр": ("length", 1), "Сантиметр": ("length", 10), "Метр": ("length", 1000),
    "Километр; тысяча метров": ("length", 1_000_000), "Дюйм (25,4 мм)": ("length", 25.4),
    "Миллиграмм": ("mass", 0.000001), "Грамм": ("mass", 0.001), "Килограмм": ("mass", 1), "Тонна": ("mass", 1000),
    "Ватт": ("power", 1), "Киловатт": ("power", 1000), "Вольт": ("voltage", 1),
    "Ампер-час (3,6 кКл)": ("charge", 1), "Оборот в минуту": ("rpm", 1), "Штука": ("count", 1),
    "Градус Цельсия": ("temp", 1), "Квадратный миллиметр": ("area", 1), "Минута": ("time", 1),
}

# как единицу пишут пользователи → (размерность, множитель)
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
# полные слова пишутся с окончаниями («сантиметров», «килограмма»), сокращения — без
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
# значение в открытом диапазоне КТРУ («≥ 100 Сантиметр»), которое дальше самой крайней границы
# в столько раз, считаем подозрительным: «стол высотой 5 метров», «молоток 50 кг»
SUSPICIOUS_FACTOR = 3


SHORT_UNITS = {("mass", 1): "кг", ("mass", 0.001): "г", ("mass", 1e-6): "мг", ("mass", 1e-9): "мкг", ("mass", 1000): "т", ("length", 1): "мм",
               ("length", 10): "см", ("length", 100): "дм", ("length", 1000): "м", ("length", 25.4): "дюйм",
               ("power", 1): "Вт", ("power", 1000): "кВт", ("voltage", 1): "В", ("charge", 1): "А·ч",
               ("charge", 0.001): "мА·ч", ("rpm", 1): "об/мин", ("count", 1): "шт", ("temp", 1): "°C"}


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
    """Число — часть кода модели/артикула: вплотную к буквам, дефисам, слэшам («П-30-950», «2930-H5», «HR2450FT»)."""
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


class NumberMatcher:
    def __init__(self, store: KtruStore):
        self.store = store

    def _num_chars(self, codes: list[str]) -> dict[str, dict]:
        """Числовые характеристики кандидатов: имя → {unit, keys, by_code: {код: values}}."""
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

    def match(self, text: str, codes: list[str]) -> tuple[list[dict], list[dict], list[list[int]], list[dict]]:
        """→ (характеристики, ошибки, спаны слов, занятых числами — их не берут текстовые характеристики,
        числа без места в КТРУ у этих товаров — для Портала / новых характеристик)."""
        chars = self._num_chars(codes)
        rank = {c: i for i, c in enumerate(codes)}
        product_words = {t["lemma"] for c in codes for t in tokenize(self.store.by_code[c]["name"])}
        found, errors, reserved, unmatched = [], [], [], []
        tokens = tokenize(text)

        for m in _mentions(text):
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
                continue  # «RUKO 108225», «П-30-950» — артикул или модель, а не значение
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
            # среди подходящих по словам — совместимые по единице
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
        # одна характеристика — одно значение (первое упоминание)
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
            # для диапазона «52-54» значение должно покрывать обе границы
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
    """Значение формально попало в открытый диапазон КТРУ, но слишком далеко от его границы."""
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
