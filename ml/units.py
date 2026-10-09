"""Числа и единицы измерения → числовые характеристики КТРУ.

«высота 110 см» → «Высота: 1000–1200 Миллиметр» (110 см = 1100 мм попадает в диапазон).

1. Находим числа: «110 см», «0,5кг», «52-54», «от 100 до 200 мм», «3 полки».
2. Определяем характеристику:
   - по словам рядом с числом («высота 110», «массой 0.5 кг», «3 полки»);
   - если слов нет — по единице, когда у кандидатов ровно одна характеристика такой
     размерности («молоток 0.5 кг» → «Масса молотка»).
3. Переводим в единицу КТРУ и ищем диапазон, в который попадает число.
Ошибки: единица не подходит («длина 2 кг»), число вне всех допустимых диапазонов.
"""
import re

from ml.ktru import KtruStore, format_value, is_text_char
from ml.text import tokenize

# единица КТРУ (ОКЕИ) → (размерность, множитель к базовой единице размерности)
KTRU_UNITS = {
    "Миллиметр": ("length", 1), "Сантиметр": ("length", 10), "Метр": ("length", 1000),
    "Километр; тысяча метров": ("length", 1_000_000), "Дюйм (25,4 мм)": ("length", 25.4),
    "Грамм": ("mass", 0.001), "Килограмм": ("mass", 1), "Тонна": ("mass", 1000),
    "Ватт": ("power", 1), "Киловатт": ("power", 1000), "Вольт": ("voltage", 1),
    "Ампер-час (3,6 кКл)": ("charge", 1), "Оборот в минуту": ("rpm", 1), "Штука": ("count", 1),
    "Градус Цельсия": ("temp", 1), "Квадратный миллиметр": ("area", 1), "Минута": ("time", 1),
}

# как единицу пишут пользователи → (размерность, множитель)
USER_UNITS = {
    "мм": ("length", 1), "миллиметр": ("length", 1), "см": ("length", 10), "сантиметр": ("length", 10),
    "дм": ("length", 100), "м": ("length", 1000), "метр": ("length", 1000), "км": ("length", 1_000_000),
    '"': ("length", 25.4), "дюйм": ("length", 25.4),
    "г": ("mass", 0.001), "гр": ("mass", 0.001), "грамм": ("mass", 0.001), "кг": ("mass", 1),
    "килограмм": ("mass", 1), "т": ("mass", 1000), "тонна": ("mass", 1000),
    "вт": ("power", 1), "ватт": ("power", 1), "квт": ("power", 1000), "киловатт": ("power", 1000),
    "в": ("voltage", 1), "вольт": ("voltage", 1),
    "ач": ("charge", 1), "а*ч": ("charge", 1), "мач": ("charge", 0.001),
    "об/мин": ("rpm", 1), "шт": ("count", 1), "штука": ("count", 1),
    "°c": ("temp", 1), "°с": ("temp", 1), "градус": ("temp", 1), "мм2": ("area", 1),
}
# полные слова пишутся с окончаниями («сантиметров», «килограмма»), сокращения — без
UNIT_WORDS = ["миллиметр", "сантиметр", "метр", "дюйм", "килограмм", "грамм", "тонн", "киловатт", "ватт",
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

    def match(self, text: str, codes: list[str]) -> tuple[list[dict], list[dict], list[list[int]]]:
        """→ (характеристики, ошибки, спаны слов, занятых числами — их не берут текстовые характеристики)."""
        chars = self._num_chars(codes)
        rank = {c: i for i, c in enumerate(codes)}
        found, errors, reserved = [], [], []
        tokens = tokenize(text)

        for m in NUMBER_RE.finditer(text):
            a = _num(m.group("a"))
            b = _num(m.group("b")) if m.group("b") else None
            unit = _user_unit(m.group("unit"))
            # слова рядом с числом в пределах фрагмента (до запятой)
            seg_start = max(text.rfind(",", 0, m.start()), text.rfind(";", 0, m.start())) + 1
            seg_end = min([i for i in (text.find(",", m.end()), text.find(";", m.end())) if i != -1] or [len(text)])
            ctx_tokens = ([t for t in tokens if seg_start <= t["span"][0] and t["span"][1] <= m.start()
                           and t["text"][0].isalpha()][-CONTEXT_BEFORE:]
                          + [t for t in tokens if m.end() <= t["span"][0] and t["span"][1] <= seg_end
                             and t["text"][0].isalpha()][:CONTEXT_AFTER])
            context = [t["lemma"] for t in ctx_tokens]
            source = text[m.start():m.end()].strip()
            span = [m.start(), m.start() + len(source)]

            scored = [(self._key_score(context, info["keys"]), name, info) for name, info in chars.items()]
            named = [x for x in scored if x[0] > 0]
            if named:
                best_score = max(x[0] for x in named)
                options = [x for x in named if x[0] == best_score]
            elif unit:  # слов нет — ищем по единице измерения
                options = [x for x in scored if KTRU_UNITS.get(x[2]["unit"], (None,))[0] == unit[0]]
                if len({x[1] for x in options}) != 1:
                    continue  # неоднозначно («2 м» — длина? ширина?) — не угадываем
            else:
                continue  # голое число без слов и единицы

            if named:
                keys = {k for x in options for k in x[2]["keys"]}
                reserved += [t["span"] for t in ctx_tokens if self._key_score([t["lemma"]], list(keys))]
            # среди подходящих по словам — совместимые по единице
            compatible = [x for x in options if unit is None or KTRU_UNITS.get(x[2]["unit"], (None,))[0] == unit[0]]
            if not compatible:
                name, info = options[0][1], options[0][2]
                errors.append({"type": "unit_mismatch", "characteristic": name, "source": source, "span": span,
                               "message": f"Единица «{m.group('unit')}» не подходит для характеристики «{name}» "
                                          f"(ожидается {info['unit'] or 'безразмерное число'})."})
                continue
            name = min(compatible, key=lambda x: min(rank[c] for c in x[2]["by_code"]))[1]
            entry, err = self._resolve(name, chars[name], a, b, unit, source, span, rank)
            if entry:
                found.append(entry)
            if err:
                errors.append(err)
        # одна характеристика — одно значение (первое упоминание)
        uniq = {}
        for f in found:
            uniq.setdefault(f["name"], f)
        return list(uniq.values()), errors, reserved

    @staticmethod
    def _resolve(name, info, a, b, unit, source, span, rank):
        codes, best = [], None
        for code, (ktru_unit, values) in sorted(info["by_code"].items(), key=lambda kv: rank[kv[0]]):
            dim, factor = KTRU_UNITS.get(ktru_unit, (None, 1))
            conv = (lambda x: x * unit[1] / factor) if unit and dim == unit[0] else (lambda x: x)
            xa = conv(a)
            xb = conv(b) if b is not None else None
            # для диапазона «52-54» значение должно покрывать обе границы
            hit = next((v for v in values if contains(v, xa) and (xb is None or contains(v, xb))), None)
            if hit:
                codes.append(code)
                if best is None:
                    user_val = _fmt(xa) + (f"–{_fmt(xb)}" if xb is not None else "")
                    best = (format_value(hit), ktru_unit, user_val)
        if best is None:
            ktru_unit, values = next(iter(info["by_code"].values()))
            allowed = ", ".join(format_value(v, ktru_unit) for v in values[:12]) + (" …" if len(values) > 12 else "")
            return None, {"type": "out_of_range", "characteristic": name, "source": source, "span": span,
                          "message": f"Значение «{source}» не входит в допустимые значения характеристики «{name}». "
                                     f"Допустимые: {allowed}."}
        value, ktru_unit, user_val = best
        return {"name": name, "value": value, "unit": ktru_unit, "user_value": f"{user_val} {ktru_unit or ''}".strip(),
                "source": source, "span": span, "method": "number", "score": 95, "codes": codes}, None
