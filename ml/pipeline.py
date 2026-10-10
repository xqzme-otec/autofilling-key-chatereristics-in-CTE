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
                       conflict, contradiction, product_not_found, invalid_input (severity=error);
                       suspicious_value, multiple_values, portal_unknown_value, invalid_answer (warning);
                       not_in_ktru (info) — признаки вне КТРУ, сохранённые как Портал / новые;
    next_step        — что спросить у пользователя (type: product | characteristic | code) или null;
    ktru, ktru_name  — итоговый код, когда он определён однозначно;
    steps            — как сужался список кодов (для отображения);
    missing_required — обязательные характеристики КТРУ оставшихся позиций, которых нет в описании;
    timings_ms       — время этапов.
"""
import re
import time

from ml.correct import Corrector
from ml.ktru import DEFAULT_PATH, KtruStore
from ml.narrow import answer_to_characteristic, missing_required, narrow
from ml.search import EMB_MODEL, Searcher
from ml.text import main_noun, normalize, pos, to_nominative, tokenize
from ml.portal import PortalMatcher, new_characteristic
from ml.units import DIMENSION_NAMES, NumberMatcher
from ml.values import GENERIC, ValueMatcher, _lemmas, _stem_eq

SEARCH_K = 50
MIN_SCORE_GAP = 0.03       # кандидаты, сильно отстающие от лидера по близости, отбрасываются
NOT_FOUND_HARD = 0.80      # ниже — товара в КТРУ точно нет
NOT_FOUND_SOFT = 0.86      # ниже — нет, если ни одно слово запроса не совпало с названиями кандидатов
MAX_CANDIDATES_OUT = 20
VALUE_ERRORS = ("nonexistent_value", "out_of_range", "unit_mismatch")
PUBLIC_CHAR_FIELDS = ("name", "value", "unit", "user_value", "source", "span", "method")
ANSWER_TYPES = ("product", "characteristic", "code")
# words that never form a feature on their own
SKIP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "NUMR", "ADVB", "PRED", "COMP"}
# attribute names: alone, without a value, they are not a feature of the product («Арт.», «Размер», «Производитель»)
ATTRIBUTE_WORDS = {"арт", "артикул", "модель", "производитель", "страна", "бренд", "марка", "количество", "размер",
                   "тип", "вид", "комплект", "комплектация", "набор", "серия", "код", "цвет", "материал", "состав",
                   "назначение", "исполнение", "гарантия", "упаковка", "описание", "параметр", "характеристика"}


def _between_digits(text: str, i: int) -> bool:
    return 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()


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
        raw, answers, input_errors = self._validate(request)
        timings, t0 = {}, time.perf_counter()

        def tick(name):
            nonlocal t0
            now = time.perf_counter()
            timings[name] = round((now - t0) * 1000, 1)
            t0 = now

        # 1. нормализация и исправления
        text = normalize(raw)
        tokens = request.get("tokens") if text == raw and isinstance(request.get("tokens"), list) else None
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
        num_chars, num_errors, reserved, unmatched_numbers = self.numbers.match(text, codes)
        text_chars, value_errors, normalizations = self.values.match(text, codes, reserved)
        corrections += normalizations
        # a value that formally fits an open KTRU range but looks wrong must not pick the final code
        suspicious = {tuple(e["span"]) for e in num_errors if e["type"] == "suspicious_value"}
        for f in num_chars:
            f["suspicious"] = tuple(f["span"]) in suspicious
        answered = []
        for a in answers:
            if a["type"] != "characteristic":
                continue
            f = answer_to_characteristic(a, codes, self.store)
            if f["codes"]:
                answered.append(f)
            else:
                input_errors.append({"type": "invalid_answer", "severity": "warning", "characteristic": a["characteristic"],
                                     "message": f"Ответ «{a['characteristic']} = {a['value']}» не подходит ни к одной "
                                                f"из найденных позиций КТРУ и не учтён."})
        names = {c["name"] for c in candidates}
        for a in answers:
            if a["type"] == "product" and a["value"] not in names:
                input_errors.append({"type": "invalid_answer", "severity": "warning",
                                     "message": f"Товар «{a['value']}» не входит в найденные варианты и не учтён."})
        answered_names = {a["name"] for a in answered}
        characteristics = answered + sorted(
            (f for f in text_chars + num_chars if f["name"] not in answered_names),
            key=lambda f: -f["score"])
        tick("values")

        # 4. сужение
        remaining, next_step, steps, conflicts = narrow(characteristics, candidates, self.store, head, answers)
        tick("narrow")

        # 5. то, что не легло на КТРУ, не выбрасываем: справочник Портала, иначе — новая характеристика
        extra_notices = []
        if not not_found and candidates:
            pool = remaining or candidates
            category = self.store.by_code[pool[0]["code"]]["category"]
            # found among search candidates, but none of the remaining positions has it: not KTRU for this product
            pool_chars = {ch["name"] for c in pool for ch in self.store.by_code[c["code"]]["characteristics"]}
            for f in characteristics:
                if f.get("origin") is None and f["method"] != "answer" and f["name"] not in pool_chars:
                    f["origin"] = "portal" if self.portal.has(category, f["name"]) else "new"
                    if f.get("user_value"):  # a number: show it as the user wrote it, KTRU ranges do not apply
                        f["value"], f["unit"] = f["source"], None
            taken = [f["span"] for f in characteristics if f.get("span")] + \
                    [e["span"] for e in value_errors + num_errors if e.get("span")] + reserved + \
                    self.values.last_consumed
            product_words = {l for c in pool for l in _lemmas(c["name"])}
            extra, extra_notices = self._extra_characteristics(text, category, taken, unmatched_numbers,
                                                                product_words, pool, characteristics)
            characteristics += extra
        tick("extra")

        errors = input_errors + value_errors + num_errors + conflicts + extra_notices
        if not_found:
            errors = [{"type": "product_not_found",
                       "message": "Не удалось уверенно определить товар в КТРУ (категории: одежда, мебель, "
                                  "инструменты). Уточните наименование товара; ниже — ближайшие по смыслу позиции."}]
            next_step = None
            errors = input_errors + errors
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
            "missing_required": [] if not_found or ktru else missing_required(remaining, characteristics, self.store),
            "timings_ms": timings,
        }

    def _extra_characteristics(self, text: str, category: str | None, taken: list[list[int]],
                               unmatched_numbers: list[dict], product_words: set,
                               pool: list[dict], characteristics_ref: list[dict] = ()) -> tuple[list[dict], list[dict]]:
        """Признаки из описания, которых нет среди характеристик КТРУ найденных товаров:
        справочник Портала («цвет синий» → «Цвет»), иначе новая характеристика
        («с противоскользящей рукояткой» → «Противоскользящая рукоятка: Да»). Не выбрасываем ничего."""
        extra, notices, taken = [], [], list(taken)

        def add(item: dict, source: str, span: list[int]) -> None:
            origin = "portal" if item.pop("portal", False) else "new"
            unknown, known = item.pop("unknown_value", False), item.pop("known_values", [])
            extra.append({**item, "source": source, "span": span, "method": origin, "score": 70 if origin == "portal" else 50,
                          "codes": [], "origin": origin})
            if unknown:
                notices.append({"type": "portal_unknown_value", "severity": "warning", "characteristic": item["name"],
                                "source": source, "span": span,
                                "message": f"Значение «{item['value']}» не встречается у характеристики «{item['name']}» "
                                           f"на Портале поставщиков (встречаются: {', '.join(known)}). Проверьте значение."})

        # numbers without a place among KTRU characteristics: «стол письменный высотой 75 см»
        for n in unmatched_numbers:
            phrase = f"{n['keyword']} {n['source']}" if n["keyword"] else n["source"]
            found = self.portal.match(phrase, category)
            if found:
                item = {**found, "portal": True}
            else:
                name = to_nominative([n["keyword"]]).capitalize() if n["keyword"] else \
                    DIMENSION_NAMES.get(n["dimension"], "Числовая характеристика")
                item = {"name": name, "value": n["value_text"], "unit": None}
            start = n["keyword_span"][0] if n["keyword_span"] else n["span"][0]
            add(item, text[start:n["span"][1]], [start, n["span"][1]])
            taken += [n["span"]] + ([n["keyword_span"]] if n["keyword_span"] else [])

        # free words not used by KTRU, numbers, errors or the product name: «с противоскользящей рукояткой»
        def is_taken(span):
            return any(a < span[1] and span[0] < b for a, b in taken)

        head_end = next((i for i, ch in enumerate(text) if ch in ",;\n" and not _between_digits(text, i)), len(text))
        groups, current = [], []
        for t in tokenize(text):
            word = t["text"]
            boundary = word in ",;.\n" and not (word in ".," and _between_digits(text, t["span"][0]))
            if current and word[0].isdigit() and not is_taken(t["span"]):
                current.append(t)  # «гарантия 3 года»: the number belongs to the phrase
                continue
            if boundary or is_taken(t["span"]) or not word[0].isalpha():
                if current:
                    groups.append(current)
                current = []
                continue
            # product-name words are skipped only in the first fragment, where the product itself is named
            in_head = t["span"][0] < head_end
            if word.isascii() or len(word) < 3 or (in_head and t["lemma"] in product_words) or pos(word) in SKIP_POS:
                if current and pos(word) not in ("PREP", "CONJ"):  # a preposition may sit inside a phrase
                    groups.append(current)
                    current = []
                continue
            current.append(t)
        if current:
            groups.append(current)

        split = []
        for g in groups:
            no_noun = all(pos(t["text"]) != "NOUN" for t in g) and not any(t["text"][0].isdigit() for t in g)
            split += [[t] for t in g] if no_noun and len(g) > 1 else [g]
        for g in split:
            if len(g) == 1 and g[0]["lemma"] in ATTRIBUTE_WORDS:
                continue  # an attribute name without a readable value
            span = [g[0]["span"][0], g[-1]["span"][1]]
            phrase = text[span[0]:span[1]]
            found = self.portal.match(phrase, category)
            if found:
                item = {**found, "portal": True}
            elif any(t["text"][0].isdigit() for t in g):  # «гарантия 3 года» → name + value
                item = new_characteristic(phrase)
            else:
                item = {"name": to_nominative([t["text"] for t in g]).capitalize(), "value": "Да", "unit": None}
            add(item, phrase, span)

        moved = [f for f in characteristics_ref if f.get("origin") in ("portal", "new") and f not in extra]
        if extra or moved:
            product = pool[0]["name"] if len({c["name"] for c in pool}) == 1 else "найденных товаров"
            listed = ", ".join(f"«{f['name']}»" for f in moved + extra)
            notices.append({"type": "not_in_ktru", "severity": "info",
                            "characteristics": [f["name"] for f in moved + extra],
                            "message": f"В КТРУ для «{product}» нет характеристик: {listed}. Они сохранены "
                                       f"(источник — справочник Портала поставщиков или новая характеристика) "
                                       f"и на выбор кода КТРУ не влияют."})
        return extra, notices

    @staticmethod
    def _validate(request: dict) -> tuple[str, list[dict], list[dict]]:
        """Bad input becomes a readable error instead of an exception."""
        errors = []
        raw = request.get("raw") if isinstance(request, dict) else None
        if not isinstance(raw, str):
            errors.append({"type": "invalid_input", "severity": "error", "message": "Поле «raw» должно быть строкой."})
            raw = ""
        answers_in = request.get("answers") if isinstance(request, dict) else None
        if answers_in is not None and not isinstance(answers_in, list):
            errors.append({"type": "invalid_answer", "severity": "warning", "message": "Поле «answers» должно быть списком."})
            answers_in = []
        answers = []
        for a in answers_in or []:
            ok = isinstance(a, dict) and isinstance(a.get("value"), str) and a.get("type", "characteristic") in ANSWER_TYPES
            if ok and a.get("type", "characteristic") == "characteristic" and not isinstance(a.get("characteristic"), str):
                ok = False
            if not ok:
                errors.append({"type": "invalid_answer", "severity": "warning",
                               "message": f"Ответ {a!r} не распознан и не учтён: нужны поля type, value "
                                          f"(и characteristic для type=characteristic)."})
                continue
            answers.append({**a, "type": a.get("type", "characteristic")})
        return raw, answers, errors

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
