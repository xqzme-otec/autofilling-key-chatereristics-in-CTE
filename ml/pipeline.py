from __future__ import annotations

"""Точка входа ML: analyze(запрос) -> готовый JSON для фронта.

Вход (dict или просто строка):
    {
      "raw": "стл офисный на колесиках, кожзам, высота 110 см",
      "tokens": [...],                       # необязательно: токены от бэкенда {text, lemma, span, known}
      "answers": [                           # необязательно: ответы на предыдущие вопросы next_step
        {"type": "product", "value": "Стул офисный"},
        {"type": "characteristic", "characteristic": "Наличие подлокотников", "value": "Да"},
        {"type": "code", "value": "31.01.11.150-00000003"}
      ]
    }

Выход:
    input            — исходный текст;
    text             — текст после нормализации и исправлений;
    corrections      — исправления (опечатки, раскладка — span в input;
                       приведение к значениям КТРУ — span в text);
    characteristics  — {name, value, unit, source, span (в text), method, confidence, origin};
                       source — фрагмент текста пользователя; origin — откуда характеристика:
                       ktru (справочник КТРУ) | portal (справочник Портала) | new (нет ни там, ни там —
                       сохраняем, не выбрасываем) | answer (выбрана пользователем из вариантов КТРУ);
    candidates       — оставшиеся коды КТРУ {code, name, score}, candidates_total — их число;
    errors           — {type, severity, message, ...}: nonexistent_value, out_of_range, unit_mismatch,
                       conflict, product_not_found (severity=error); suspicious_value (severity=warning);
    next_step        — что спросить у пользователя (type: product | characteristic | code) или null;
    ktru, ktru_name  — итоговый код, когда он определён однозначно;
    steps            — как сужался список кодов (для отображения);
    timings_ms       — время этапов.
"""
import re
import time

from ml.correct import Corrector
from ml.ktru import DEFAULT_PATH, KtruStore
from ml.narrow import answer_to_characteristic, narrow
from ml.search import EMB_MODEL, Searcher
from ml.text import main_noun, normalize, tokenize
from ml.portal import PortalMatcher, new_characteristic
from ml.units import NumberMatcher
from ml.values import GENERIC, ValueMatcher, _lemmas, _stem_eq

SEARCH_K = 50
MIN_SCORE_GAP = 0.03       # кандидаты, сильно отстающие от лидера по близости, отбрасываются
NOT_FOUND_HARD = 0.80      # ниже — товара в КТРУ точно нет
NOT_FOUND_SOFT = 0.86      # ниже — нет, если ни одно слово запроса не совпало с названиями кандидатов
MAX_CANDIDATES_OUT = 20
VALUE_ERRORS = ("nonexistent_value", "out_of_range", "unit_mismatch")
PUBLIC_CHAR_FIELDS = ("name", "value", "unit", "user_value", "source", "span", "method")


class Pipeline:
    def __init__(self, ktru_path=DEFAULT_PATH, model_name: str = EMB_MODEL, rerank: bool = False):
        self.store = KtruStore(ktru_path)
        self.corrector = Corrector(self.store)
        self.searcher = Searcher(self.store, model_name, rerank)
        self.values = ValueMatcher(self.store, self.searcher)
        self.numbers = NumberMatcher(self.store)
        self.portal = PortalMatcher()

    def analyze(self, request: dict | str) -> dict:
        if isinstance(request, str):
            request = {"raw": request}
        raw = request["raw"]
        answers = request.get("answers") or []
        timings, t0 = {}, time.perf_counter()

        def tick(name):
            nonlocal t0
            now = time.perf_counter()
            timings[name] = round((now - t0) * 1000, 1)
            t0 = now

        # 1. нормализация и исправления
        text = normalize(raw)
        tokens = request.get("tokens") if text == raw else None
        text, corrections = self.corrector.fix_layout(text)
        if corrections or tokens is None:
            tokens = tokenize(text)
        text, typos = self.corrector.correct(text, tokens)
        corrections += typos
        tick("correct")

        # 2. поиск кандидатов: весь текст + первый фрагмент (обычно это название товара)
        head = text.split(",")[0]
        candidates = self.searcher.search(text, k=SEARCH_K, extra_queries=[head])
        not_found = self._not_found(head, candidates)
        if candidates and not self.searcher.reranker:
            best = candidates[0]["score"]
            candidates = [c for c in candidates if best - c["score"] <= MIN_SCORE_GAP]
        codes = [c["code"] for c in candidates]
        tick("search")

        # 3. характеристики: текстовые значения и числа
        num_chars, num_errors, reserved = self.numbers.match(text, codes)
        text_chars, value_errors, normalizations = self.values.match(text, codes, reserved)
        corrections += normalizations
        answered = [answer_to_characteristic(a, codes, self.store)
                    for a in answers if a.get("type", "characteristic") == "characteristic"]
        answered_names = {a["name"] for a in answered}
        characteristics = answered + sorted(
            (f for f in text_chars + num_chars if f["name"] not in answered_names),
            key=lambda f: -f["score"])
        tick("values")

        # 4. сужение
        remaining, next_step, steps, conflicts = narrow(characteristics, candidates, self.store, head, answers)
        tick("narrow")

        # 5. то, что не легло на КТРУ, не выбрасываем: справочник Портала, иначе — новая характеристика
        if not not_found:
            category = self.store.by_code[(remaining or candidates)[0]["code"]]["category"] if candidates else None
            taken = [f["span"] for f in characteristics if f.get("span")] + \
                    [e["span"] for e in value_errors + num_errors if e.get("span")] + reserved
            characteristics += self._extra_characteristics(text, category, taken)
        tick("extra")

        errors = value_errors + num_errors + conflicts
        if not_found:
            errors = [{"type": "product_not_found",
                       "message": "Не удалось уверенно определить товар в КТРУ (категории: одежда, мебель, "
                                  "инструменты). Уточните наименование товара; ниже — ближайшие по смыслу позиции."}]
            next_step = None
        # значения проверяем по допустимым у найденных позиций; если сам товар определён неточно
        # (наименование не совпало с текстом и не выбрано пользователем), это не ошибка пользователя
        # «несуществующее значение» достаточно проверить по товарам с тем же главным словом («стул»);
        # диапазоны и единицы — только у точно определённой позиции («сверло по металлу» ≠ «Сверло центровочное»)
        exact = any(st["stage"].startswith(("Наименование:", "Выбран товар", "Выбран код")) for st in steps)
        by_noun = exact or any(st["stage"].startswith("Наименование содержит") for st in steps)
        for e in errors:
            confirmed = by_noun if e["type"] == "nonexistent_value" else exact
            if not confirmed and e["type"] in VALUE_ERRORS:
                e["severity"] = "warning"
                e["message"] += " (позиция КТРУ определена неточно — уточните наименование товара)"
            e.setdefault("severity", "error")
        ktru = remaining[0] if len(remaining) == 1 and not not_found else None

        return {
            "input": raw,
            "text": text,
            "corrections": corrections,
            "characteristics": [self._public(f) for f in characteristics if not f.get("conflict")],
            "candidates": remaining[:MAX_CANDIDATES_OUT],
            "candidates_total": len(remaining),
            "errors": errors,
            "next_step": next_step,
            "ktru": ktru["code"] if ktru else None,
            "ktru_name": ktru["name"] if ktru else None,
            "steps": steps,
            "timings_ms": timings,
        }

    def _extra_characteristics(self, text: str, category: str | None, taken: list[list[int]]) -> list[dict]:
        """Фрагменты описания (кроме первого — это название товара), не занятые КТРУ:
        «цвет синий» → Портал, «противоскользящая рукоятка» → новая характеристика."""
        extra = []
        segments = list(re.finditer(r"(?:[^,;\n.]|\.(?=\d))+", text))
        for m in segments[1:]:
            seg = m.group().strip()
            start = m.start() + len(m.group()) - len(m.group().lstrip())
            span = [start, start + len(seg)]
            if len(seg) < 3 or not any(c.isalpha() for c in seg) \
                    or any(a < span[1] and span[0] < b for a, b in taken):
                continue
            found = self.portal.match(seg, category)
            origin, item = ("portal", found) if found else ("new", new_characteristic(seg))
            extra.append({**item, "source": seg, "span": span, "method": origin,
                          "score": 70 if found else 50, "codes": [], "origin": origin})
        return extra

    def _not_found(self, head: str, candidates: list[dict]) -> bool:
        if not candidates or self.searcher.reranker:
            return not candidates
        top = candidates[0]["score"]
        if top < NOT_FOUND_HARD:
            return True
        # главное существительное запроса должно встречаться в названиях кандидатов:
        # «ноутбук игровой» ≠ «Стол игровой» (совпало прилагательное, а не товар)
        noun = main_noun(head)
        query = [noun] if noun else [l for l in _lemmas(head) if l not in GENERIC and len(l) > 2 and l.isalpha()]
        names = {l for c in candidates[:10] for l in _lemmas(c["name"])}
        name_hit = any(_stem_eq(q, n) for q in query for n in names)
        return not name_hit and top < NOT_FOUND_SOFT

    @staticmethod
    def _public(f: dict) -> dict:
        out = {k: f[k] for k in PUBLIC_CHAR_FIELDS if k in f}
        out["confidence"] = round(f["score"] / 100, 2)
        # происхождение (иерархия источников): КТРУ → Портал → новая; answer — выбрано пользователем из вариантов КТРУ
        out["origin"] = f.get("origin") or ("answer" if f["method"] == "answer" else "ktru")
        return out
