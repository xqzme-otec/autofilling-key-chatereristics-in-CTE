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
                       suspicious_value, multiple_values, portal_unknown_value, invalid_answer,
                       missing_required (warning);
                       not_in_ktru (info) — признаки вне КТРУ, сохранённые как Портал / новые;
    next_step        — что спросить у пользователя (type: product | characteristic | code) или null;
    ktru, ktru_name  — итоговый код, когда он определён однозначно;
    steps            — как сужался список кодов (для отображения);
    missing_required — обязательные характеристики КТРУ, которых нет в описании: у определённого кода —
                       его собственные, пока кодов несколько — обязательные у всех оставшихся позиций;
                       о них же предупреждение missing_required в errors;
    timings_ms       — время этапов.
"""
import re
import time

from ml.correct import Corrector
from ml.ktru import DEFAULT_PATH, KtruStore
from ml.narrow import answer_to_characteristic, missing_required, narrow, rerank_and_question
from ml.search import EMB_MODEL, Searcher
from ml.text import is_known, lemma, main_noun, normalize, pos, to_nominative, tokenize
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


# names of the maker or the model, not features of the product: their value goes to a brand notice
BRAND_ATTRIBUTES = {"арт", "артикул", "модель", "производитель", "бренд", "марка", "серия", "код"}
QUOTED = re.compile(r"[«\"“„]([^«»\"“”„\n]{1,40})[»\"”“]")


def _clause_start(text: str, i: int) -> bool:
    before = text[:i].rstrip()
    return not before or before[-1] in ",;.:\n("


def _brand_word(word: str, text: str, start: int, vocab) -> bool:
    """A maker or model name, not a feature: «Bosch», «CH-868», «ШХА» in a mixed-case text, «Юнитекс» capitalized
    in the middle of a clause. A word of the KTRU vocabulary is a value written with a capital letter
    («Материал Полипропилен», «ОФИСНАЯ ТУМБА»), not a name."""
    if not any(ch.isalpha() for ch in word):
        return False
    if word.isascii():
        return len(word) >= 2
    if word.lower() in vocab or lemma(word) in vocab:
        return False
    if len(word) >= 2 and word.isupper() and not text.isupper():
        return True
    return len(word) >= 3 and word[0].isupper() and word[1:].islower() and not _clause_start(text, start)


def _code_part(token: dict, end: int, text: str, vocab) -> bool:
    """The token continues a model name: glued to it («CH-868», «PH2»), or an article number of 5+ digits; not the
    first number of a size («DeWalt 22х450мм»)."""
    word, gap = token["text"], token["span"][0] - end
    if gap == 0:
        return word[0].isdigit() or word in "-/." or _brand_word(word, text, token["span"][0], vocab)
    return gap == 1 and (word.isdigit() and len(word) >= 5 or _brand_word(word, text, token["span"][0], vocab))


VOWELS = set("аеёиоуыэюя")
MIN_KNOWN_TRIGRAMS = 0.6  # «ыврпаоы» 0.0, «ваываыв» 0.29; rare real words: «гварнери» 0.75, «однослойная» 0.91


def _trigrams(words) -> set:
    out = set()
    for w in words:
        w = f"^{w}$"
        out.update(w[i:i + 3] for i in range(len(w) - 2))
    return out


def _gibberish(word: str, trigrams: set, vocab) -> bool:
    """A lowercase Cyrillic letter string that is no Russian word: no vowels, starts with «ы», a run of five
    consonants, or letter triples unseen in the KTRU vocabulary. Unknown but plausible words are not gibberish."""
    if not re.fullmatch(r"[а-яё]+", word) or is_known(word) or word in vocab:
        return False
    padded = f"^{word}$"
    triples = [padded[i:i + 3] for i in range(len(padded) - 2)]
    known_share = sum(t in trigrams for t in triples) / len(triples)
    return not VOWELS & set(word) or word[0] in "ыъь" or bool(re.search(r"[^аеёиоуыэюя]{5,}", word)) \
        or known_share < MIN_KNOWN_TRIGRAMS


def _morph_case(word: str) -> str | None:
    from ml.text import _morph
    return _morph.parse(word.lower())[0].tag.case


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
        self.trigrams = _trigrams(w for w in self.store.vocab if re.fullmatch(r"[а-яё]{3,}", w))

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
            # «куртка» matched to «Вид санитарной одежды: Куртка» of a dropped candidate: the product itself
            characteristics = [f for f in characteristics if not (
                f.get("origin") is None and f["method"] != "answer" and f["name"] not in pool_chars
                and product_intent and len((f.get("source") or "").split()) == 1
                and main_noun(f.get("source") or "") == product_intent)]
            for f in characteristics:
                if f.get("origin") is None and f["method"] != "answer" and f["name"] not in pool_chars \
                        and not f.get("used_for_narrowing"):
                    f["origin"] = "portal" if self.portal.has(category, f["name"], product_intent) else "new"
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
        # required KTRU characteristics the user did not give: a hint, the code is still determined without them
        required = [] if not_found else missing_required(remaining, characteristics, self.store)
        if required:
            listed = ", ".join(f"«{name}»" for name in required)
            errors.append({"type": "missing_required", "severity": "warning", "characteristics": required,
                           "message": f"Вы не ввели обязательную характеристику товара: {listed}. Добавьте её в описание."
                                      if len(required) == 1 else
                                      f"Вы не ввели обязательные характеристики товара: {listed}. Добавьте их в описание."})

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
            "missing_required": required,
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

        def overlaps(span, spans):
            return any(a < span[1] and span[0] < b for a, b in spans)

        upstream = list(taken)  # spans of KTRU characteristics, numbers and errors found before this step

        # maker and model names: «Bosch», «"Олимп"», «Юнитекс», «CH-868» — reported, not turned into features
        brands: list[list[int]] = []
        for m in QUOTED.finditer(text):
            if not overlaps([m.start(), m.end()], taken):
                brands.append([m.start(1), m.end(1)])
        tokens = tokenize(text)
        for i, t in enumerate(tokens):
            if overlaps(t["span"], taken + brands) or not _brand_word(t["text"], text, t["span"][0], self.store.vocab):
                continue
            end = t["span"][1]
            for nxt in tokens[i + 1:]:  # «CH-868», «ШХА-900(40)»: the code continues the name
                if overlaps(nxt["span"], taken) or not _code_part(nxt, end, text, self.store.vocab):
                    break
                end = nxt["span"][1]
            while end > t["span"][1] and text[end - 1] in "-/.":
                end -= 1
            brands.append([t["span"][0], end])
        taken += brands

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
            if overlaps(n["span"], brands):
                continue  # a model code: «CH-868»
            if n["keyword_span"] and re.search(r"(?:^|\s)по\s+$", text[:n["keyword_span"][0]]):
                # «сверло по металлу 6 мм»: «металлу» is the purpose, not the name of the number
                n = {**n, "keyword": None, "keyword_span": None}
            elif n["keyword"] and _lemmas(n["keyword"])[:1] and \
                    (_lemmas(n["keyword"])[0] == product_intent or _lemmas(n["keyword"])[0] in product_words):
                # «футболка мужская 2 л»: the product is not the name of the number («Футболка: 2 л»)
                n = {**n, "keyword": None, "keyword_span": None}
            if n["keyword_span"]:
                # «длина рукава 60 см», «частота вращения 3000 об/мин»: the name is the whole phrase before the
                # number, not its last word («Рукава: 60 см» + «Длина: Да»)
                start = self._name_start(text, n["keyword_span"][0], taken + brands, product_words, product_intent)
                if start < n["keyword_span"][0]:
                    n = {**n, "keyword": text[start:n["keyword_span"][1]], "keyword_span": [start, n["keyword_span"][1]]}
            phrase = f"{n['keyword']} {n['source']}" if n["keyword"] else n["source"]
            found = self.portal.match(phrase, category, product_intent)
            if found and n["keyword"] and len(_lemmas(found["name"])) < len(_lemmas(n["keyword"])):
                found = None  # «Длина: рукава 60 см»: the portal knows only a shorter name, keep the user's one
            if found and n.get("synthetic"):
                # a size or a dimension named by us: its format («120x60x75 см», «XL») is not compared
                # with the portal's free-text examples
                found.pop("unknown_value", None)
                found.pop("known_values", None)
            if found:
                item = {**found, "portal": True}
            else:
                head, *rest = (n["keyword"] or "").split()  or [""]
                name = " ".join([to_nominative([head]).capitalize(), *rest]) if n["keyword"] else \
                    DIMENSION_NAMES.get(n["dimension"], "Числовая характеристика")
                item = {"name": name, "value": n["value_text"], "unit": None}
            start = n["keyword_span"][0] if n["keyword_span"] else n["span"][0]
            add(item, text[start:n["span"][1]], [start, n["span"][1]])
            taken += [n["span"]] + ([n["keyword_span"]] if n["keyword_span"] else [])

        # free words not used by KTRU, numbers, errors or the product name: «с противоскользящей рукояткой»
        def is_taken(span):
            return any(a < span[1] and span[0] < b for a, b in taken)

        for t in tokenize(text):
            portal_value = self.portal.match(t["text"], category, product_intent)
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
            product_word = in_head and (t["lemma"] in product_words or t["lemma"] == product_intent)
            if word.isascii() or len(word) < 3 or product_word or pos(word) in SKIP_POS:
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
        consumed: list[list[int]] = []
        unreadable: list[list[int]] = []
        for g in split:
            span = [g[0]["span"][0], g[-1]["span"][1]]
            phrase = text[span[0]:span[1]]
            if overlaps(span, consumed):
                continue  # the value of a characteristic name met earlier
            if self._unreadable(g):
                if unreadable and not text[unreadable[-1][1]:span[0]].strip():
                    unreadable[-1][1] = span[1]  # «ппп ффф ыыы»: one fragment
                else:
                    unreadable.append(span)
                continue
            if self._part_of_matched_name(g, span, text, upstream):
                continue
            purpose = re.search(r"(?:^|\s)(по)\s+$", text[:span[0]])
            if purpose and pos(g[0]["text"]) == "NOUN" and not any(t["text"][0].isdigit() for t in g):
                # «сверло по металлу»: the purpose of the product, a yes/no feature on the portal or «Назначение»
                start = purpose.start(1)
                found = self.portal.match(text[start:span[1]], category, product_intent)
                item = {**found, "portal": True} if found else \
                    {"name": "Назначение", "value": text[start:span[1]].lower(), "unit": None}
                add(item, text[start:span[1]], [start, span[1]])
                continue
            found = self.portal.match(phrase, category, product_intent)
            if g[0]["lemma"] in ATTRIBUTE_WORDS and not found and not any(t["text"][0].isdigit() for t in g):
                used = self._attribute_without_value(g, text, span, category, product_intent, brands, taken, add,
                                                     notices)
                if used:
                    consumed.append(used)
                continue
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
                # «в собранном виде Нет»: the answer after the phrase is its value, not a lost word
                answer = re.match(r"[\s:—-]*(да|нет)\b", text[span[1]:], re.IGNORECASE)
                item = {"name": to_nominative([t["text"] for t in g]).capitalize(),
                        "value": answer.group(1).capitalize() if answer else "Да", "unit": None}
            add(item, phrase, span)

        for a, b in unreadable:
            notices.append({"type": "unreadable_fragment", "severity": "warning", "source": text[a:b], "span": [a, b],
                            "message": f"Не удалось распознать «{text[a:b]}». Поясните, что имелось в виду: "
                                       "пока этот фрагмент не учтён."})
        if brands:
            names = list(dict.fromkeys(text[a:b].strip() for a, b in sorted(brands) if text[a:b].strip()))
            if names:
                notices.append({"type": "brand_or_model", "severity": "info", "characteristics": names,
                                "message": "Бренд, модель или артикул не относятся к характеристикам КТРУ и в подборе "
                                           f"кода не участвуют: {', '.join(f'«{n}»' for n in names)}."})
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

    def _unreadable(self, group: list[dict]) -> bool:
        """Every word is gibberish even after the typo correction: «ыврпаоы», «ппп ффф». Capitalized words are
        names, compounds («серый-черный») count by their parts."""
        words = [t["text"] for t in group if t["text"][0].isalpha()]
        return bool(words) and all(
            w.islower() and all(_gibberish(part, self.trigrams, self.store.vocab) for part in w.split("-") if part)
            for w in words)

    def _attribute_without_value(self, group: list[dict], text: str, span: list[int], category: str | None,
                                 product: str | None, brands: list[list[int]], taken: list[list[int]], add,
                                 notices: list[dict]) -> list[int] | None:
        """«материал изготовления HSS», «цвет - синий», «производитель Bosch»: the name of a characteristic, its
        value is the rest of the clause (without parentheses). A value already used by another characteristic
        leaves nothing to do; no value at all — the user is asked for it. Returns the span used as the value."""
        end = next((i for i in range(span[1], len(text)) if text[i] in ",;\n"), len(text))
        lemmas = [t["lemma"] for t in group]
        entry = next((name for name_lemmas, name, _ in self.portal.by_category.get(category, [])
                      if name_lemmas == lemmas), None)
        name = entry or text[span[0]:span[1]][:1].upper() + text[span[0]:span[1]][1:]
        if lemmas[0] in BRAND_ATTRIBUTES:
            if not any(span[1] <= b[0] < end for b in brands):
                brands.append(span)  # «Производитель» without a name: reported with the brands
            return None
        depth, rest = 0, []
        for t in tokenize(text[span[1]:end]):
            a, b = t["span"][0] + span[1], t["span"][1] + span[1]
            depth += t["text"].count("(") - t["text"].count(")")
            if depth <= 0 and t["text"][0].isalnum():
                rest.append([a, b])
        free = [s for s in rest if not any(x < s[1] and s[0] < y for x, y in taken)
                or any(x <= s[0] and s[1] <= y for x, y in brands)]
        if rest and not free:
            return None  # «материал изготовления сталь» with «сталь» taken by «Материал бойка»
        if not free:
            notices.append({"type": "missing_value", "severity": "warning", "characteristic": name,
                            "source": text[span[0]:span[1]], "span": span,
                            "message": f"Для «{name}» не указано значение — уточните его."})
            return None
        value_span = [free[0][0], free[-1][1]]
        brands[:] = [b for b in brands if not (value_span[0] <= b[0] and b[1] <= value_span[1])]
        value = text[value_span[0]:value_span[1]]
        found = self.portal.match(f"{name} {value}", category, product)
        item = {**found, "portal": True} if found else {"name": name, "value": value, "unit": None,
                                                        "portal": bool(entry)}
        add(item, text[span[0]:value_span[1]], [span[0], value_span[1]])
        return value_span

    @staticmethod
    def _name_start(text: str, keyword_start: int, taken: list[list[int]], product_words: set,
                    product: str | None) -> int:
        """Where the name of a number starts: up to three words before its keyword in the same clause — nouns,
        adjectives and a preposition inside («нагрузка на полку»); not the product, not a taken word."""
        words = [t for t in tokenize(text[:keyword_start]) if t["span"][1] <= keyword_start]
        start, count = keyword_start, 0
        for t in reversed(words):
            word = t["text"]
            if not word[0].isalpha() or any(a < t["span"][1] and t["span"][0] < b for a, b in taken) \
                    or t["lemma"] in product_words or t["lemma"] == product or count >= 3:
                break
            kind = pos(word)
            if kind == "PREP" and word.lower() in ("на", "для", "по", "до", "от"):
                continue  # kept only if a word before it joins the name
            if kind not in ("NOUN", "ADJF", "PRTF"):
                break
            start, count = t["span"][0], count + 1
        return start

    @staticmethod
    def _part_of_matched_name(group: list[dict], span: list[int], text: str, upstream: list[list[int]]) -> bool:
        """«Угол заточки 135 градус» with «угол … 135 градус» matched: «заточки» is a word of that name, not a
        feature. A short noun group right before a matched span, or a genitive right after one."""
        after = len(text[span[1]:]) - len(text[span[1]:].lstrip())
        if len(group) <= 2 and all(pos(t["text"]) == "NOUN" for t in group) \
                and any(a == span[1] + after for a, _ in upstream):
            return True  # «толщина каркаса 20 мм»; an adjective makes it a feature: «противоскользящая рукоятка 30 см»
        before = text[:span[0]].rstrip()
        return any(b == len(before) for _, b in upstream) and all(
            pos(t["text"]) in ("NOUN", "ADJF", "PRTF") and _morph_case(t["text"]) == "gent" for t in group)

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
