from __future__ import annotations

"""ML entry point: analyze(request) -> ready JSON for the frontend.

Input (a dict or just a string):
    {
      "raw": "стл офисный на колесиках, кожзам, высота 110 см",
      "tokens": [...],                       # optional: tokens from the backend {text, lemma, span, known}
      "answers": [                           # optional: answers to previous next_step questions
        {"type": "product", "value": "Стул офисный"},
        {"type": "characteristic", "characteristic": "Наличие подлокотников", "value": "Да"},
        {"type": "code", "value": "31.01.11.150-00000003"}
      ]
    }

Output:
    input            — the original text;
    text             — the text after normalization and corrections;
    corrections      — corrections (typos, keyboard layout — span in input;
                       normalization to KTRU values — span in text);
    characteristics  — {name, value, unit, source, span (in text), method, confidence, origin};
                       source — fragment of the user text; origin — where the characteristic comes from:
                       ktru (KTRU reference) | portal (portal reference) | new (in neither —
                       kept, never dropped) | answer (chosen by the user from KTRU options);
    category         — category of the found products (одежда | мебель | инструменты) or null;
    candidates       — remaining KTRU codes {code, name, score}, candidates_total — their count;
    errors           — {type, severity, message, ...}: nonexistent_value, out_of_range, unit_mismatch,
                       conflict, contradiction, product_not_found, invalid_input (severity=error);
                       suspicious_value, multiple_values, portal_unknown_value, invalid_answer (warning);
                       not_in_ktru (info) — features outside KTRU, kept as portal / new;
    next_step        — what to ask the user (type: product | characteristic | code) or null;
    ktru, ktru_name  — the final code once it is determined unambiguously;
    steps            — how the code list was narrowed (for display);
    missing_required — required KTRU characteristics of the remaining positions missing from the description;
    timings_ms       — stage timings.
"""
import re
import time

from ml.correct import Corrector
from ml.ktru import DEFAULT_PATH, KtruStore
from ml.narrow import answer_to_characteristic, missing_required, narrow, rerank_and_question
from ml.search import EMB_MODEL, Searcher
from ml.text import main_noun, normalize, pos, to_nominative, tokenize
from ml.portal import PortalMatcher, new_characteristic
from ml.sizes import SizeMatcher
from ml.units import DIMENSION_NAMES, NumberMatcher
from ml.values import GENERIC, ValueMatcher, _lemmas, _stem_eq

SEARCH_K = 50
MIN_SCORE_GAP = 0.03       # candidates far behind the leader by similarity are dropped
NOT_FOUND_HARD = 0.80      # below this the product is surely not in KTRU
NOT_FOUND_SOFT = 0.86      # below this — not found if no query word matches candidate names
MAX_CANDIDATES_OUT = 20
VALUE_ERRORS = ("nonexistent_value", "out_of_range", "unit_mismatch")
PUBLIC_CHAR_FIELDS = ("name", "value", "unit", "user_value", "source", "span", "method")
ANSWER_TYPES = ("product", "characteristic", "code")
# words that never form a feature on their own
SKIP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "NUMR", "ADVB", "PRED", "COMP"}
# attribute names: alone, without a value, they are not a feature of the product («Арт.», «Размер», «Производитель»)
ATTRIBUTE_WORDS = {"арт", "артикул", "модель", "производитель", "страна", "бренд", "марка", "количество", "размер",
                   "тип", "вид", "комплект", "комплектация", "набор", "серия", "код", "цвет", "материал", "состав",
                   "назначение", "исполнение", "гарантия", "упаковка", "описание", "параметр", "характеристика",
                   "габарит"}


def _between_digits(text: str, i: int) -> bool:
    return 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()


class Pipeline:
    def __init__(self, ktru_path=DEFAULT_PATH, model_name: str = EMB_MODEL, rerank: bool = False):
        self.store = KtruStore(ktru_path)
        self.corrector = Corrector(self.store)
        self.searcher = Searcher(self.store, model_name, rerank)
        self.values = ValueMatcher(self.store, self.searcher)
        self.numbers = NumberMatcher(self.store)
        self.sizes = SizeMatcher(self.store)
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

        # 1. normalization and corrections
        text = normalize(raw)
        tokens = request.get("tokens") if text == raw and isinstance(request.get("tokens"), list) else None
        text, corrections = self.corrector.fix_layout(text)
        if corrections or tokens is None:
            tokens = tokenize(text)
        text, typos = self.corrector.correct(text, tokens)
        corrections += typos
        tick("correct")

        # 2. candidate search: the whole text + the first fragment (usually the product name)
        head = text.split(",")[0]
        product_intent = main_noun(head)
        candidates = self.searcher.search(text, k=SEARCH_K, extra_queries=[head])
        not_found = self._not_found(head, candidates)
        if candidates and not self.searcher.reranker:
            best = candidates[0]["score"]
            candidates = [c for c in candidates if best - c["score"] <= MIN_SCORE_GAP]
        codes = [c["code"] for c in candidates]
        tick("search")

        # 3. characteristics: text values and numbers
        # clothing sizes first («р. 52-54», «XL»): their numbers and letters are not taken by other matchers
        size_chars, size_errors, size_reserved, unmatched_sizes = self.sizes.match(text, codes)
        num_chars, num_errors, reserved, unmatched_numbers = self.numbers.match(text, codes, size_reserved)
        num_chars = size_chars + [f for f in num_chars if f["name"] not in {s["name"] for s in size_chars}]
        num_errors = size_errors + num_errors
        reserved += size_reserved
        unmatched_numbers += unmatched_sizes
        text_chars, value_errors, normalizations = self.values.match(text, codes, reserved)
        text_chars = [f for f in text_chars if f["name"] not in {s["name"] for s in size_chars}]
        corrections += normalizations
        # a value that formally fits an open KTRU range but looks wrong must not pick the final code
        # (keyed by characteristic too: the numbers of «120x60x75 см» share one span)
        suspicious = {(e["characteristic"], tuple(e["span"])) for e in num_errors if e["type"] == "suspicious_value"}
        for f in num_chars:
            f["suspicious"] = (f["name"], tuple(f["span"])) in suspicious
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

        # 4. narrowing
        remaining, next_step, steps, conflicts = narrow(characteristics, candidates, self.store, head, answers)
        tick("narrow")

        # 5. what did not fit KTRU is not dropped: the portal reference, otherwise a new characteristic
        extra_notices, signals, category = [], [], None
        if not not_found and candidates:
            pool = remaining or candidates
            category = self.store.by_code[pool[0]["code"]]["category"]
            # found among search candidates, but none of the remaining positions has it: not KTRU for this product
            pool_chars = {ch["name"] for c in pool for ch in self.store.by_code[c["code"]]["characteristics"]}
            for f in characteristics:
                if f.get("origin") is None and f["method"] != "answer" and f["name"] not in pool_chars \
                        and not f.get("used_for_narrowing"):
                    # a dimension of a dropped candidate («Ширина стеллажа» for a chair) keeps only its axis
                    f["name"] = f.get("axis") or f["name"]
                    f["origin"] = "portal" if self.portal.has(category, f["name"]) else "new"
                    if f.get("user_value"):  # a number: show it as the user wrote it, KTRU ranges do not apply
                        f["value"], f["unit"] = f.get("value_text") or f["source"], None
            taken = [f["span"] for f in characteristics if f.get("span")] + \
                    [e["span"] for e in value_errors + num_errors if e.get("span")] + reserved + \
                    self.values.last_consumed
            product_words = {l for c in pool for l in _lemmas(c["name"])}
            extra, extra_notices, signals = self._extra_characteristics(text, category, taken, unmatched_numbers,
                                                                         product_words, pool, characteristics,
                                                                         product_intent)
            characteristics += extra
            remaining, next_step, soft_step = rerank_and_question(remaining, characteristics, signals, self.store,
                                                                   self.portal, self.searcher, answers)
            if soft_step:
                steps.append(soft_step)
            else:
                for signal in signals:
                    signal["used_for_reranking"] = False
        tick("extra")

        errors = input_errors + value_errors + num_errors + conflicts + extra_notices
        if not_found:
            errors = [{"type": "product_not_found",
                       "message": "Не удалось уверенно определить товар в КТРУ (категории: одежда, мебель, "
                                  "инструменты). Уточните наименование товара; ниже — ближайшие по смыслу позиции."}]
            next_step = None
            errors = input_errors + errors
        # values are checked against those allowed for the found positions; if the product itself is uncertain
        # (the name did not match the text and was not chosen by the user), it is not the user's error.
        # A «nonexistent value» only needs products with the same head noun («стул»);
        # ranges and units — only an exactly determined position («сверло по металлу» ≠ «Сверло центровочное»)
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
            "product_intent": product_intent,
            "category": category,
            "subtype_context": signals,
            "retrieval_count": len(candidates),
            "compatible_count": next((s["count"] for s in steps[1:] if s["stage"].startswith("Наименование")), len(candidates)),
            "remaining_count": len(remaining),
            "candidates": remaining[:MAX_CANDIDATES_OUT],
            "candidates_total": len(remaining),
            "errors": errors,
            "next_step": next_step,
            "next_question": next_step,
            "ktru": ktru["code"] if ktru else None,
            "ktru_name": ktru["name"] if ktru else None,
            "steps": steps,
            "missing_required": [] if not_found or ktru else missing_required(remaining, characteristics, self.store),
            "timings_ms": timings,
        }

    def _extra_characteristics(self, text: str, category: str | None, taken: list[list[int]],
                               unmatched_numbers: list[dict], product_words: set,
                               pool: list[dict], characteristics_ref: list[dict] = (),
                               product_intent: str | None = None) -> tuple[list[dict], list[dict], list[dict]]:
        """Features from the description that are not among the KTRU characteristics of the found products:
        the portal reference («цвет синий» → «Цвет»), otherwise a new characteristic
        («с противоскользящей рукояткой» → «Противоскользящая рукоятка: Да»). Nothing is dropped."""
        extra, notices, signals, taken = [], [], [], list(taken)

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
            if found and n.get("synthetic"):
                # a size or a dimension named by us: its format («120x60x75 см», «XL») is not compared
                # with the portal's free-text examples
                found.pop("unknown_value", None)
                found.pop("known_values", None)
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

        for t in tokenize(text):
            portal_value = self.portal.match(t["text"], category)
            value_attribute = portal_value and portal_value["name"].lower().startswith(("цвет", "материал"))
            if not is_taken(t["span"]) and pos(t["text"]) in ("ADJF", "PRTF") \
                    and not value_attribute and self.portal.is_subtype(category, product_intent, t["lemma"]):
                signals.append({"value": to_nominative([t["text"]]).capitalize(), "user_text": t["text"],
                                "span": t["span"], "source": "portal", "used_for_narrowing": False,
                                "used_for_reranking": True})
                taken.append(t["span"])

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
            adjective = len(g) == 1 and pos(g[0]["text"]) in ("ADJF", "PRTF")
            value_attribute = found and found["name"].lower().startswith(("цвет", "материал"))
            subtype = adjective and not value_attribute and self.portal.is_subtype(category, product_intent, g[0]["lemma"])
            if adjective and (subtype or not found):
                signals.append({"value": to_nominative([g[0]["text"]]).capitalize(), "user_text": phrase,
                                "span": span, "source": "portal" if subtype else "user",
                                "used_for_narrowing": False, "used_for_reranking": True})
                continue
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
            listed = ", ".join(f"«{f['name']}»" for f in moved + extra if not f.get("used_for_narrowing"))
            if not listed:
                return extra, notices, signals
            notices.append({"type": "not_in_ktru", "severity": "info",
                            "characteristics": [f["name"] for f in moved + extra],
                            "message": f"Для «{product}» сохранены дополнительные признаки: {listed}. "
                                       "Они не исключают коды КТРУ."})
        return extra, notices, signals

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
        # the head noun of the query must occur in candidate names:
        # «ноутбук игровой» ≠ «Стол игровой» (an adjective matched, not the product)
        noun = main_noun(head)
        query = [noun] if noun else [l for l in _lemmas(head) if l not in GENERIC and len(l) > 2 and l.isalpha()]
        names = {l for c in candidates[:10] for l in _lemmas(c["name"])}
        name_hit = any(_stem_eq(q, n) for q in query for n in names)
        return not name_hit and top < NOT_FOUND_SOFT

    @staticmethod
    def _public(f: dict) -> dict:
        out = {k: f[k] for k in PUBLIC_CHAR_FIELDS if k in f}
        out["confidence"] = round(f["score"] / 100, 2)
        # origin (source hierarchy): KTRU → portal → new; answer — chosen by the user from KTRU options
        out["origin"] = f.get("origin") or ("answer" if f["method"] == "answer" else "ktru")
        out["user_text"] = out.get("source")
        out["source"] = "user" if out["origin"] in ("new", "answer") else out["origin"]
        out["used_for_narrowing"] = bool(f.get("used_for_narrowing"))
        out["used_for_reranking"] = bool(f.get("used_for_reranking"))
        return out
