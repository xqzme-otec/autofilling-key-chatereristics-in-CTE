from __future__ import annotations

"""Перевод слов пользователя в текстовые характеристики КТРУ.

«на колесиках» → «Наличие колесных опор: Да», «кожзам» → «Материал обивки: Искусственная кожа».

Текст режется на фрагменты по запятым/точкам, каждый фрагмент сопоставляется
с характеристиками кандидатов КТРУ:
  - Да/Нет характеристики («Наличие подлокотников») — по ключевым словам названия,
    «без …» / «нет …» перед ключевым словом даёт «Нет»;
  - списочные значения («Ткань», «Сетка») — точное/нечёткое (rapidfuzz) совпадение лемм
    с учётом синонимов и обобщений («хлопок» → «Натуральная»), «без капюшона» → «Отсутствует»,
    эмбеддинги — запасной вариант.
Ошибка «несуществующее значение» — когда фрагмент называет характеристику
(«материал бойка …»), а слово рядом не подходит ни к одному её допустимому значению.

Числовые характеристики обрабатывает ml/units.py.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from ml.ktru import KtruStore, is_bool_char, is_text_char
from ml.text import main_noun, tokenize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYNONYMS_PATH = DATA_DIR / "synonyms.json"
ALIASES_PATH = DATA_DIR / "value_aliases.json"
GENERIC = {"наличие", "функция", "тип", "элемент", "вид", "количество", "покрытие", "материал", "конструкция",
           "товар", "изделие", "наименование", "признак", "особенность", "для", "с", "и", "в", "на", "из"}
NEGATIONS = {"без", "нет", "не", "отсутствовать"}
NEGATIVE_VALUES = ("отсутств", "без ", "нет")
FUZZY_MIN = 85
EMB_MIN = 0.88
EMB_MARGIN = 0.02
STEM = 5
WINDOW = 2  # сколько слов после названия характеристики считать её значением


@dataclass
class CharInfo:
    name: str
    is_bool: bool
    keys: list[str] = field(default_factory=list)          # ключевые леммы названия характеристики
    values: dict[str, set] = field(default_factory=dict)   # значение -> коды КТРУ, где оно допустимо
    value_lemmas: dict[str, str] = field(default_factory=dict)


@dataclass
class Segment:
    text: str
    full: str                    # весь текст — для вырезания source по спанам
    start: int
    lemmas: list[str] = field(default_factory=list)        # леммы с подставленными синонимами
    owner: list[int] = field(default_factory=list)         # номер токена, из которого получена лемма
    spans: list[list[int]] = field(default_factory=list)   # спаны токенов в тексте
    from_synonym: list[bool] = field(default_factory=list)

    def source(self, lemma_idx: set) -> tuple[str, list[int]]:
        toks = sorted({self.owner[i] for i in lemma_idx})
        start, end = self.spans[toks[0]][0], self.spans[toks[-1]][1]
        return self.full[start:end], [start, end]


def _load(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def load_synonyms() -> dict[str, list[str]]:
    return {k: v.split() for k, v in _load(SYNONYMS_PATH).items()}


def _stem_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    return len(a) >= STEM and len(b) >= STEM and a[:STEM] == b[:STEM]


def _lemmas(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalnum()]


class ValueMatcher:
    def __init__(self, store: KtruStore, searcher=None):
        self.store = store
        self.searcher = searcher  # для эмбеддингов; без него работают синонимы и fuzzy
        self.synonyms = load_synonyms()
        self.aliases = _load(ALIASES_PATH)
        self._emb_cache: dict[str, np.ndarray] = {}

    # ---------- подготовка ----------

    def _chars_for(self, codes: list[str]) -> dict[str, CharInfo]:
        chars: dict[str, CharInfo] = {}
        name_words: dict[str, set] = {}
        for code in codes:
            item = self.store.by_code[code]
            item_words = set(_lemmas(item["name"]))
            for ch in item["characteristics"]:
                if not is_text_char(ch):
                    continue
                name_words.setdefault(ch["name"], set()).update(item_words)
                info = chars.setdefault(ch["name"], CharInfo(ch["name"], is_bool_char(ch)))
                for v in ch["values"]:
                    if "text" in v:
                        info.values.setdefault(v["text"], set()).add(code)
                        info.value_lemmas[v["text"]] = " ".join(_lemmas(v["text"]))
        for info in chars.values():
            # слова из названия товара не ключевые: иначе «стул» включает «Поворотная конструкция стула: Да»
            info.keys = [l for l in _lemmas(info.name) if l not in GENERIC and l not in name_words[info.name]]
        # в скольких характеристиках кандидатов (включая числовые) встречается ключевое слово:
        # «рукоять» есть в «Длина рукоятки», «Материал рукояти»… — по нему одному Да/Нет не решаем
        all_names = {ch["name"] for c in codes for ch in self.store.by_code[c]["characteristics"]}
        self._key_df = {}
        for name in all_names:
            for st in {l[:STEM] for l in _lemmas(name)}:  # по основе: «рукоятка» и «рукоять» — одно слово
                self._key_df[st] = self._key_df.get(st, 0) + 1
        # «Женская», «Женское», «Женский» у разных позиций — одно значение: объединяем коды по лемме
        for info in chars.values():
            by_lemma: dict[str, set] = {}
            for value, vlem in info.value_lemmas.items():
                by_lemma.setdefault(vlem, set()).update(info.values[value])
            for value, vlem in info.value_lemmas.items():
                info.values[value] = by_lemma[vlem]
        return chars

    def _segments(self, text: str, reserved: list[list[int]] = ()) -> list[Segment]:
        segs = []
        taken = {tuple(s) for s in reserved}
        for m in re.finditer(r"[^,;.\n]+", text):
            toks = [t for t in tokenize(m.group()) if t["text"][0].isalnum()
                    and (m.start() + t["span"][0], m.start() + t["span"][1]) not in taken]
            if not toks:
                continue
            seg = Segment(m.group().strip(), text, m.start())
            for ti, t in enumerate(toks):
                seg.spans.append([m.start() + t["span"][0], m.start() + t["span"][1]])
                syn = self.synonyms.get(t["lemma"])
                for l in syn or [t["lemma"]]:
                    seg.lemmas.append(l)
                    seg.owner.append(ti)
                    seg.from_synonym.append(syn is not None and syn != [t["lemma"]])
            segs.append(seg)
        return segs

    @staticmethod
    def _key_hits(lemmas: list[str], keys: list[str]) -> list[int]:
        return [i for i, l in enumerate(lemmas) if any(_stem_eq(l, k) for k in keys)]

    # ---------- сопоставление ----------

    def _match_enum(self, seg: Segment, info: CharInfo) -> tuple[str, int, str, set] | None:
        """→ (значение, оценка, способ, номера использованных лемм) или None."""
        lemmas, best = seg.lemmas, None
        key_hits = self._key_hits(lemmas, info.keys)
        for value, vlem in info.value_lemmas.items():
            n = len(vlem.split())
            grams = [" ".join(lemmas[i:i + n]) for i in range(len(lemmas) - n + 1)]
            if vlem in grams:
                i = grams.index(vlem)
                return value, 100, "exact", set(range(i, i + n))
            # «без капюшона» → «Тип капюшона: Отсутствует»
            if value.lower().startswith(NEGATIVE_VALUES) and key_hits:
                neg = [i for i in range(len(lemmas)) if lemmas[i] in NEGATIONS and any(0 < k - i <= 2 for k in key_hits)]
                if neg:
                    best = (value, 95, "negation", {neg[0], next(k for k in key_hits if k > neg[0])})
                    continue
            # обобщения: «хлопок» → «Тип ткани: Натуральная»
            alias = [i for i, l in enumerate(lemmas) if l in self.aliases.get(vlem, ()) and l not in NEGATIONS]
            if alias:
                if best is None or best[1] < 88:
                    best = (value, 88, "alias", {alias[0]})
                continue
            if n == 1:  # «нитриловые» ~ «Нитрил»
                stem_hits = [i for i, l in enumerate(lemmas) if _stem_eq(l, vlem)]
                if stem_hits:
                    if best is None or best[1] < FUZZY_MIN:
                        best = (value, FUZZY_MIN, "stem", {stem_hits[0]})
                    continue
            if len(vlem) < 5 or not grams:  # короткие (ЛДСП, МДФ) — только точно
                continue
            _, score, i = process.extractOne(vlem, grams, scorer=fuzz.ratio)
            if score >= FUZZY_MIN and (best is None or score > best[1]):
                best = (value, int(score), "fuzzy", set(range(i, i + n)))
        return best

    def _match_bool(self, seg: Segment, info: CharInfo) -> tuple[str, set] | None:
        """Да/Нет по ключевым словам: нужно совпадение хотя бы половины ключей
        («на колесиках» ~ «колесных опор» 1 из 2 — да; «утепленная» ~ «отстегивающейся
        утепляющей подкладки» 1 из 3 — нет)."""
        lemmas = seg.lemmas
        if not info.keys:
            return None
        hits = self._key_hits(lemmas, info.keys)
        matched = {k for k in info.keys if any(_stem_eq(lemmas[i], k) for i in hits)}
        if not hits or len(matched) * 2 < len(info.keys):
            return None
        if len(matched) < len(info.keys) and any(self._key_df.get(k[:STEM], 0) > 1 for k in matched):
            return None  # частичное совпадение по неразличающему слову
        i = hits[0]
        neg = [j for j in range(max(0, i - 2), i) if lemmas[j] in NEGATIONS]
        used = set(hits) | set(neg)
        if not neg and i > 0 and lemmas[i - 1] in ("с", "на", "со"):  # для читаемого source: «на колесиках»
            used.add(i - 1)
        return ("Нет" if neg else "Да"), used

    def match(self, text: str, codes: list[str], reserved: list[list[int]] = ()) -> tuple[list[dict], list[dict], list[dict]]:
        """→ (характеристики, ошибки, нормализации для corrections).
        reserved — спаны слов, уже занятых числовыми характеристиками."""
        chars = self._chars_for(codes)
        rank = {code: i for i, code in enumerate(codes)}
        product_words = {l for c in codes for l in _lemmas(self.store.by_code[c]["name"])}
        noun = main_noun(text.split(",")[0])
        found: dict[str, dict] = {}
        errors: list[dict] = []

        for seg_no, seg in enumerate(self._segments(text, reserved)):
            matches = []
            for info in chars.values():
                if info.is_bool:
                    m = self._match_bool(seg, info)
                    if m is None:
                        continue
                    value, used = m
                    if value not in info.values:
                        continue  # «без подлокотников», а у товаров бывает только «Да» — решит сужение
                    score, method = 90, "keyword"
                else:
                    m = self._match_enum(seg, info)
                    if m is None:
                        continue
                    value, score, method, used = m
                    if method in ("exact", "stem", "fuzzy") and any(seg.from_synonym[i] for i in used):
                        method = "synonym"
                    if noun and all(_stem_eq(seg.lemmas[i], noun) for i in used):
                        method = "name"  # «Вид товара = Шкаф» — повторяет сам товар, кандидатов не отсекает
                named = any(i not in used for i in self._key_hits(seg.lemmas, info.keys))
                best_rank = min(rank[c] for c in info.values[value])
                matches.append((named, score, -best_rank, info, value, method, used))

            # одно слово — одной характеристике: «на металлическом каркасе» — «Тип каркаса»
            # (каркас назван явно), а не «Вид материала спинки»; затем по оценке и рангу товара
            taken: set = set()
            accepted = []
            for mt in sorted(matches, key=lambda m: m[:3], reverse=True):
                if mt[6] & taken:
                    continue
                taken |= mt[6]
                accepted.append(mt)

            accepted, seg_errors = self._check_named(seg, chars, accepted, product_words)
            errors += seg_errors

            for named, score, _, info, value, method, used in accepted:
                prev = found.get(info.name)
                if prev is None or score > prev["score"]:
                    src, span = seg.source(used)
                    found[info.name] = {"name": info.name, "value": value, "unit": None, "source": src, "span": span,
                                        "method": method, "score": score, "codes": sorted(info.values[value])}

            if not accepted and not seg_errors and seg_no > 0:  # первый фрагмент — название товара
                emb = self._match_embedding(seg, chars)
                if emb:
                    info, value, score = emb
                    found.setdefault(info.name, {"name": info.name, "value": value, "unit": None, "source": seg.text,
                                                 "span": [seg.start, seg.start + len(seg.text)], "method": "embedding",
                                                 "score": score, "codes": sorted(info.values[value])})

        normalizations = [
            {"from": f["source"], "to": f["value"].lower(), "span": f["span"],
             "reason": f"приведено к значению КТРУ («{f['name']}»)"}
            for f in found.values()
            if f["method"] in ("synonym", "alias", "stem", "fuzzy", "embedding")
            and f["source"].lower() != f["value"].lower()]
        return list(found.values()), errors, normalizations

    def _check_named(self, seg: Segment, chars: dict[str, CharInfo], accepted: list,
                     product_words: set) -> tuple[list, list]:
        """Характеристика названа («материал бойка …»), но её значение не нашлось. Слово после
        названия — значение именно для неё, даже если оно подошло к другой характеристике
        («деревянный» → «Материал рукояти»): такое совпадение снимаем и сообщаем об ошибке."""
        lemmas, errors, reported = seg.lemmas, [], {}
        matched_names = {m[3].name for m in accepted}
        used_by = {i: m for m in accepted for i in m[6]}
        all_keys = {k for info in chars.values() for k in info.keys}
        for info in chars.values():
            if info.is_bool or info.name in matched_names:
                continue
            hits = self._key_hits(lemmas, info.keys)
            if not hits or any(lemmas[j] in NEGATIONS for j in range(max(0, hits[0] - 2), hits[0])):
                continue
            window = {j for h in hits for j in range(h + 1, h + WINDOW + 1)} | {hits[0] - 1}
            window = sorted(j for j in window if 0 <= j < len(lemmas) and j not in hits
                            and lemmas[j].isalpha() and lemmas[j] not in GENERIC and lemmas[j] not in NEGATIONS
                            and lemmas[j] not in product_words
                            and not any(_stem_eq(lemmas[j], k) for k in all_keys)
                            and not (j in used_by and used_by[j][0]))  # слово занято явно названной характеристикой
            if not window:
                continue  # характеристика названа без значения — не ошибка
            stolen = {id(used_by[j]) for j in window if j in used_by}
            accepted = [m for m in accepted if id(m) not in stolen]
            overlap = sum(any(_stem_eq(l, x) for x in lemmas) for l in _lemmas(info.name))
            key = tuple(window)
            if key in reported and reported[key][0] >= overlap:
                continue
            src, span = seg.source(set(window))
            err = self._not_allowed(info, src, span)
            hint = process.extractOne(" ".join(lemmas[j] for j in window), info.value_lemmas, scorer=fuzz.ratio)
            if hint and hint[1] >= 60:
                err["suggestion"] = hint[2]
            reported[key] = (overlap, err)
        return accepted, [err for _, err in reported.values()]

    def _match_embedding(self, seg: Segment, chars: dict[str, CharInfo]):
        if self.searcher is None or len(seg.lemmas) > 4:
            return None
        options = [(info, v) for info in chars.values() if not info.is_bool for v in info.values]
        if not options:
            return None
        q = self.searcher.encode([seg.text])[0]
        keys = [f"{info.name}: {v}" for info, v in options]
        new = [k for k in keys if k not in self._emb_cache]
        if new:
            self._emb_cache.update(zip(new, self.searcher.encode(new, query=False)))
        sims = np.stack([self._emb_cache[k] for k in keys]) @ q
        order = sims.argsort()[::-1]
        top = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        if top >= EMB_MIN and top - second >= EMB_MARGIN:
            info, v = options[order[0]]
            return info, v, round(top * 100)
        return None

    @staticmethod
    def _not_allowed(info: CharInfo, value: str, span: list[int]) -> dict:
        allowed = ", ".join(sorted(info.values))
        return {"type": "nonexistent_value", "characteristic": info.name, "value": value, "source": value, "span": span,
                "message": f"Значение «{value}» недопустимо для характеристики «{info.name}». Допустимые значения: {allowed}."}
