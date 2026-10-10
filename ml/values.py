from __future__ import annotations

"""User words → text KTRU characteristics.

«на колесиках» → «Наличие колесных опор: Да», «кожзам» → «Материал обивки: Искусственная кожа».

The text is split into fragments by commas/dots; each fragment is matched
against the characteristics of the KTRU candidates:
  - yes/no characteristics («Наличие подлокотников») — by the key words of the name,
    «без …» / «нет …» before a key word gives «Нет»;
  - list values («Ткань», «Сетка») — exact/fuzzy (rapidfuzz) lemma match
    with synonyms and generalizations («хлопок» → «Натуральная»), «без капюшона» → «Отсутствует»,
    embeddings as a fallback.
A «nonexistent value» error is raised when a fragment names a characteristic
(«материал бойка …») and the word next to it fits none of its allowed values.

Numeric characteristics are handled by ml/units.py.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from ml.ktru import KtruStore, is_bool_char, is_text_char
from ml.sizes import is_letter_size_char
from ml.text import main_noun, pos, tokenize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYNONYMS_PATH = DATA_DIR / "synonyms.json"
ALIASES_PATH = DATA_DIR / "value_aliases.json"
GENERIC = {"наличие", "функция", "тип", "элемент", "вид", "количество", "покрытие", "материал", "конструкция",
           "товар", "изделие", "наименование", "признак", "особенность", "максимальный", "минимальный",
           "номинальный", "общий", "для", "с", "и", "в", "на", "из", "по"}
PREPOSITIONS = {"по", "для", "с", "из", "от", "на", "без"}
NEGATIONS = {"без", "нет", "не", "отсутствовать"}
NEGATIVE_VALUES = ("отсутств", "без ", "нет")
FUZZY_MIN = 85
PARTIAL_SCORE = 80
EMB_MIN = 0.89
EMB_MARGIN = 0.02
STEM = 5
WINDOW = 2  # how many words after a characteristic name are taken as its value


@dataclass
class CharInfo:
    name: str
    is_bool: bool
    keys: list[str] = field(default_factory=list)          # key lemmas of the characteristic name
    values: dict[str, set] = field(default_factory=dict)   # value -> KTRU codes where it is allowed
    value_lemmas: dict[str, str] = field(default_factory=dict)


@dataclass
class Segment:
    text: str
    full: str                    # the whole text — to cut `source` by spans
    start: int
    lemmas: list[str] = field(default_factory=list)        # lemmas with synonyms substituted
    owner: list[int] = field(default_factory=list)         # index of the token the lemma comes from
    spans: list[list[int]] = field(default_factory=list)   # token spans in the text
    from_synonym: list[bool] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)            # lemma as written, aligned with `lemmas`

    def surface(self, lemma_idx: int) -> str:
        start, end = self.spans[self.owner[lemma_idx]]
        return self.full[start:end]

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
        self.searcher = searcher  # for embeddings; without it synonyms and fuzzy matching still work
        self.synonyms = load_synonyms()
        self.aliases = _load(ALIASES_PATH)
        self._emb_cache: dict[str, np.ndarray] = {}

    # ---------- preparation ----------

    def _chars_for(self, codes: list[str]) -> dict[str, CharInfo]:
        chars: dict[str, CharInfo] = {}
        name_words: dict[str, set] = {}
        for code in codes:
            item = self.store.by_code[code]
            item_words = set(_lemmas(item["name"]))
            for ch in item["characteristics"]:
                # letter sizes (S, M, XL) are matched by ml/sizes.py: here «2 м» would become «Размер: М»
                if not is_text_char(ch) or is_letter_size_char(ch):
                    continue
                name_words.setdefault(ch["name"], set()).update(item_words)
                info = chars.setdefault(ch["name"], CharInfo(ch["name"], is_bool_char(ch)))
                for v in ch["values"]:
                    if "text" in v:
                        info.values.setdefault(v["text"], set()).add(code)
                        info.value_lemmas[v["text"]] = " ".join(_lemmas(v["text"]))
        for info in chars.values():
            # product-name words are not key words: otherwise «стул» triggers «Поворотная конструкция стула: Да»
            info.keys = [l for l in _lemmas(info.name) if l not in GENERIC and l not in name_words[info.name]]
        # in how many characteristics of the candidates (numeric included) a key word occurs:
        # «рукоять» is in «Длина рукоятки», «Материал рукояти»… — it alone cannot decide yes/no
        all_names = {ch["name"] for c in codes for ch in self.store.by_code[c]["characteristics"]}
        self._key_df = {}
        for name in all_names:
            for st in {l[:STEM] for l in _lemmas(name)}:  # by stem: «рукоятка» and «рукоять» are one word
                self._key_df[st] = self._key_df.get(st, 0) + 1
        # «Женская», «Женское», «Женский» in different positions are one value: merge codes by lemma
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
        for m in re.finditer(r"(?:[^,;.\n]|(?<=\d)[.,](?=\d))+", text):
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
                    seg.raw.append(t["lemma"])
            segs.append(seg)
        return segs

    @staticmethod
    def _key_hits(lemmas: list[str], keys: list[str]) -> list[int]:
        return [i for i, l in enumerate(lemmas) if any(_stem_eq(l, k) for k in keys)]

    # ---------- matching ----------

    def _exact_values(self, seg: Segment, info: CharInfo) -> list[tuple[str, set]]:
        """All values of a characteristic written exactly in the fragment, on disjoint words."""
        out, used = [], set()
        for value, vlem in sorted(info.value_lemmas.items(), key=lambda kv: -len(kv[1].split())):
            n = len(vlem.split())
            if vlem.replace(" ", "").isdigit():
                continue
            for i in range(len(seg.lemmas) - n + 1):
                idx = set(range(i, i + n))
                if " ".join(seg.lemmas[i:i + n]) == vlem and not idx & used:
                    if all(vlem != info.value_lemmas[v] for v, _ in out):
                        out.append((value, idx))
                    used |= idx
        return out

    def _match_enum(self, seg: Segment, info: CharInfo) -> tuple[str, int, str, set] | None:
        """→ (value, score, method, indices of used lemmas) or None."""
        lemmas, best = seg.lemmas, None
        key_hits = self._key_hits(lemmas, info.keys)
        for value, vlem in info.value_lemmas.items():
            n = len(vlem.split())
            grams = [" ".join(lemmas[i:i + n]) for i in range(len(lemmas) - n + 1)]
            if vlem.replace(" ", "").replace(".", "").replace(",", "").isdigit() and not key_hits and len(lemmas) > n:
                continue  # a number is a value («Ростовая группа: 5», «Размер: 48») only if the characteristic
                                    # is named («ростовая группа 5») or the number is a separate fragment («куртка, 48»)
            if vlem in grams:
                i = grams.index(vlem)
                return value, 100, "exact", set(range(i, i + n))
            raw_grams = [" ".join(seg.raw[i:i + n]) for i in range(len(seg.raw) - n + 1)]
            if vlem in raw_grams:  # the word as written is the value («лето» = «Лето», though a synonym maps it to «летний»)
                i = raw_grams.index(vlem)
                return value, 100, "exact", set(range(i, i + n))
            # «без капюшона» → «Тип капюшона: Отсутствует»
            if value.lower().startswith(NEGATIVE_VALUES) and key_hits:
                neg = [i for i in range(len(lemmas)) if lemmas[i] in NEGATIONS and any(0 < k - i <= 2 for k in key_hits)]
                if neg:
                    best = (value, 95, "negation", {neg[0], next(k for k in key_hits if k > neg[0])})
                    continue
            # generalizations: «хлопок» → «Тип ткани: Натуральная»
            alias = [i for i, l in enumerate(lemmas) if l in self.aliases.get(vlem, ()) and l not in NEGATIONS]
            if alias:
                if best is None or best[1] < 88:
                    best = (value, 88, "alias", {alias[0]})
                continue
            # «По бетону» ~ «бетон»; but «Без обивки» is an absence, «без» must not be stripped
            core = vlem.split()[1] if n == 2 and vlem.split()[0] in PREPOSITIONS - {"без"} else (vlem if n == 1 else None)
            if core:  # «нитриловые» ~ «Нитрил», «бетон» ~ «По бетону»
                stem_hits = [i for i, l in enumerate(lemmas) if _stem_eq(l, core)]
                if stem_hits:
                    if best is None or best[1] < FUZZY_MIN:
                        best = (value, FUZZY_MIN, "stem", {stem_hits[0]})
                    continue
            head = vlem.split()[0]
            if n >= 2 and head not in PREPOSITIONS and len(head) >= 4:  # broader word for a specific value
                # the whole first word must match: «стекло» ~ «Стекло закаленное», but «металлический» ≠ «Металлобоксы»
                head_hits = [i for i, l in enumerate(lemmas) if l == head]
                if head_hits and (best is None or best[1] < PARTIAL_SCORE):
                    best = (value, PARTIAL_SCORE, "partial", {head_hits[0]})
            if len(vlem) < 5 or not grams:  # short values (ЛДСП, МДФ) — exact only
                continue
            _, score, i = process.extractOne(vlem, grams, scorer=fuzz.ratio)
            if score >= FUZZY_MIN and (best is None or score > best[1]):
                best = (value, int(score), "fuzzy", set(range(i, i + n)))
        return best

    def _match_bool(self, seg: Segment, info: CharInfo) -> tuple[str, set] | None:
        """Yes/no by key words: at least half of the keys must match
        («на колесиках» ~ «колесных опор» 1 of 2 — yes; «утепленная» ~ «отстегивающейся
        утепляющей подкладки» 1 of 3 — no)."""
        lemmas = seg.lemmas
        if not info.keys:
            return None
        hits = self._key_hits(lemmas, info.keys)
        matched = {k for k in info.keys if any(_stem_eq(lemmas[i], k) for i in hits)}
        if not hits or len(matched) * 2 < len(info.keys):
            return None
        if len(matched) < len(info.keys) and any(self._key_df.get(k[:STEM], 0) > 1 for k in matched):
            return None  # partial match on a non-distinctive word
        i = hits[0]
        if len(matched) < len(info.keys) and i + 1 < len(lemmas) and lemmas[i + 1][:1].isdigit():
            return None  # «высотой 5 метров» is a size, not «Регулировка по высоте: Да»
        neg = [j for j in range(max(0, i - 2), i) if lemmas[j] in NEGATIONS]
        used = set(hits) | set(neg)
        if not neg and i > 0 and lemmas[i - 1] in ("с", "на", "со"):  # for a readable source: «на колесиках»
            used.add(i - 1)
        # the same feature both with and without negation
        signs = {any(lemmas[j] in NEGATIONS for j in range(max(0, h - 2), h)) for h in hits}
        self._bool_conflict = len(signs) > 1
        return ("Нет" if neg else "Да"), used

    _bool_conflict = False

    def match(self, text: str, codes: list[str], reserved: list[list[int]] = ()) -> tuple[list[dict], list[dict], list[dict]]:
        """→ (characteristics, errors, normalizations for corrections).
        reserved — spans of words already taken by numeric characteristics."""
        chars = self._chars_for(codes)
        rank = {code: i for i, code in enumerate(codes)}
        product_words = {l for c in codes for l in _lemmas(self.store.by_code[c]["name"])}
        noun = main_noun(text.split(",")[0])
        found: dict[str, dict] = {}
        errors: list[dict] = []
        self.last_consumed = []  # spans of words used by KTRU matches and errors (for features outside KTRU)

        for seg_no, seg in enumerate(self._segments(text, reserved)):
            matches = []
            for info in chars.values():
                if info.is_bool:
                    m = self._match_bool(seg, info)
                    if m is None:
                        continue
                    value, used = m
                    if self._bool_conflict:
                        src, span = seg.source(set(self._key_hits(seg.lemmas, info.keys)) | used)
                        errors.append({"type": "contradiction", "characteristic": info.name, "source": seg.text,
                                       "span": [seg.start, seg.start + len(seg.text)],
                                       "message": f"Противоречие: в описании указано и наличие, и отсутствие "
                                                  f"(«{info.name}»). Оставьте один вариант."})
                        continue
                    if value not in info.values:
                        continue  # «без подлокотников» but the products only allow «Да» — narrowing decides
                    score, method = 90, "keyword"
                else:
                    exact = self._exact_values(seg, info)
                    if len(exact) > 1:
                        names = ", ".join(f"«{v}»" for v, _ in exact)
                        errors.append({"type": "multiple_values", "severity": "warning", "characteristic": info.name,
                                       "source": seg.text, "span": [seg.start, seg.start + len(seg.text)],
                                       "message": f"Для характеристики «{info.name}» указано несколько значений: "
                                                  f"{names}. Если это не опечатка, уточните, какое верно."})
                    m = self._match_enum(seg, info)
                    if m is None:
                        continue
                    value, score, method, used = m
                    if method in ("exact", "stem", "fuzzy") and any(seg.from_synonym[i] for i in used):
                        method = "synonym"
                    if noun and all(_stem_eq(seg.lemmas[i], noun) for i in used):
                        method = "name"  # «Вид товара = Шкаф» repeats the product itself and does not cut candidates
                named = any(i not in used for i in self._key_hits(seg.lemmas, info.keys))
                best_rank = min(rank[c] for c in info.values[value])
                # ties: the more general characteristic («Вид материала обивки», not «…обивки спинки»)
                matches.append((named, score, -len(info.keys), -best_rank, info, value, method, used))

            # one word — one characteristic: «на металлическом каркасе» is «Тип каркаса»
            # (the frame is named explicitly), not «Вид материала спинки»; then by score and product rank
            taken: set = set()
            accepted = []
            for mt in sorted(matches, key=lambda m: m[:4], reverse=True):
                if mt[7] & taken:
                    continue
                taken |= mt[7]
                accepted.append(mt)
            accepted = [(m[0], m[1], m[3], m[4], m[5], m[6], m[7]) for m in accepted]  # drop the tie-break field

            accepted, seg_errors = self._check_named(seg, chars, accepted, product_words)
            errors += seg_errors

            for named, score, _, info, value, method, used in accepted:
                key_idx = set(self._key_hits(seg.lemmas, info.keys)) | used
                self.last_consumed.append(seg.source(key_idx)[1])
                prev = found.get(info.name)
                if prev is None or score > prev["score"]:
                    src, span = seg.source(used)
                    codes_ok = set(info.values[value])
                    if method in ("partial", "stem"):  # broad material fits its qualified KTRU values
                        head = info.value_lemmas[value].split()[0]
                        codes_ok = set().union(*(info.values[v] for v, vl in info.value_lemmas.items()
                                                 if _stem_eq(vl.split()[0], head)))
                    found[info.name] = {"name": info.name, "value": value, "unit": None, "source": src, "span": span,
                                        "method": method, "score": score, "codes": sorted(codes_ok)}

            if not accepted and not seg_errors and seg_no > 0:  # the first fragment is the product name
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
        """A characteristic is named («материал бойка …») but its value was not found. The word after
        the name is the value for exactly this characteristic, even if it fit another one
        («деревянный» → «Материал рукояти»): such a match is removed and an error is reported."""
        lemmas, errors, reported = seg.lemmas, [], {}
        matched_names = {m[3].name for m in accepted}
        used_by = {i: m for m in accepted for i in m[6]}
        all_keys = {k for info in chars.values() for k in info.keys}
        for info in chars.values():
            if info.is_bool or info.name in matched_names:
                continue
            hits = self._key_hits(lemmas, info.keys)
            matched = {k for k in info.keys if any(_stem_eq(lemmas[i], k) for i in hits)}
            numeric = all(vl.replace(" ", "").isdigit() for vl in info.value_lemmas.values())
            # numeric lists («Российский размер»: 40, 42, …) — one key word is enough: «размер 300»
            if not hits or (not numeric and len(matched) * 2 <= len(info.keys) and len(info.keys) > 1):
                continue  # «комплектация» does not name «Количество аккумуляторов в комплекте»
            if any(lemmas[j] in NEGATIONS for j in range(max(0, hits[0] - 2), hits[0])):
                continue
            # «цвет сиденья синий», «толщина сидения 16 мм» — another attribute precedes the key word,
            # so it is about another characteristic, not «Вид материала сиденья»
            # an adjective right before the key word is its value («деревянным бойком»); a noun is another attribute
            name_lemmas = set(_lemmas(info.name))
            if any(lemmas[j].isalpha() and lemmas[j] not in name_lemmas | GENERIC | product_words | PREPOSITIONS
                   and (j < hits[0] - 1 or pos(seg.surface(j)) == "NOUN")
                   for j in range(0, hits[0])):
                continue
            window = {j for h in hits for j in range(h + 1, h + WINDOW + 1)} | {hits[0] - 1}
            window = sorted(j for j in window if 0 <= j < len(lemmas) and j not in hits
                            and (lemmas[j].isalpha() or (numeric and lemmas[j].replace(".", "").isdigit()))
                            and lemmas[j] not in GENERIC and lemmas[j] not in NEGATIONS
                            and lemmas[j] not in product_words
                            and not any(_stem_eq(lemmas[j], k) for k in all_keys)
                            and not (j in used_by and used_by[j][0]))  # the word is taken by an explicitly named characteristic
            if not window:
                continue  # a characteristic named without a value is not an error
            stolen = {id(used_by[j]) for j in window if j in used_by}
            accepted = [m for m in accepted if id(m) not in stolen]
            overlap = sum(any(_stem_eq(l, x) for x in lemmas) for l in _lemmas(info.name))
            key = tuple(window)
            if key in reported and reported[key][0] >= overlap:
                continue
            src, span = seg.source(set(window))
            err = self._not_allowed(info, src, span)
            self.last_consumed.append(seg.source(set(window) | set(hits))[1])
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
