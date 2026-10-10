from __future__ import annotations

"""Clothing sizes: letter sizes («XL», «р. M», «S-M», «2XL», «L/48-50») and Russian sizes after a size word
(«р. 52-54», «р. 96-100/182-188»).

KTRU has two kinds of size characteristics:
  - letter values — «Размер»: S, М, L, XL, XXL (radiation protection aprons and bibs) — matched directly;
  - the Russian size — «Российский размер» (and «Размер» of jackets): 38, 40, …, 80 and «44-46».
    A letter size is converted by the table below. One letter covers two Russian sizes (XL ≈ 52–54), so every KTRU
    value inside that range fits: the codes are the union, the shown value is «52–54», method «size_table».
    A Russian size written by the user next to the letter («L/48-50») is used instead of the table.

Assumption: the table is the common Russian retail chart (men: XL = 52–54, women: XL = 48–50).
The women's table is used when the text says «женский», otherwise the men's one (unisex and workwear follow it).
Children's clothing is sized by height, so letters are not converted when the text says «детский».
"""
import re

from ml.ktru import KtruStore, format_value
from ml.text import tokenize

SIZE_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "4XL", "5XL", "6XL", "7XL"]
# letter → Russian sizes (lowest, highest)
SIZE_TABLE = {
    "men": {"XXS": (42, 44), "XS": (44, 46), "S": (46, 48), "M": (48, 50), "L": (50, 52), "XL": (52, 54),
            "XXL": (54, 56), "XXXL": (56, 58), "4XL": (58, 60), "5XL": (60, 62), "6XL": (62, 64), "7XL": (64, 66)},
    "women": {"XXS": (38, 40), "XS": (40, 42), "S": (42, 44), "M": (44, 46), "L": (46, 48), "XL": (48, 50),
              "XXL": (50, 52), "XXXL": (52, 54), "4XL": (54, 56), "5XL": (56, 58), "6XL": (58, 60), "7XL": (60, 62)},
}
RU_SIZE_NAMES = ("Российский размер", "Размер")
SIZE_NAME = "Размер"  # name for a size without a KTRU characteristic (portal / new level)
# cyrillic look-alikes typed instead of latin letters: «ХL», «М»
LOOKALIKE = str.maketrans("ХхМм", "XxMm")
_LETTER = r"(?:[2-7](?=[XxХх]))?[XxХх]{0,4}[SsMmLlМм]"
# «L-образный», «M-12» are not sizes; «L/48-50» (letter / Russian size) and «L/17» are
LETTER_RE = re.compile(rf"(?<![\w\-/])({_LETTER})(?:\s*[-/]\s*({_LETTER}))?(?![\w]|-\w|/[^\W\d_])")
# «L/48-50», «XL 54-56», «L (48-50)»: the Russian size written next to the letter
RU_AFTER_LETTER = re.compile(r"\s*[/(]?\s*(\d{2})(?:\s*-\s*(\d{2}))?\)?(?![\w.,]?\d)(?!\s*[а-яa-z])", re.IGNORECASE)
SIZE_WORD = r"(?:(?:российск[а-я]*\s+)?размер[а-я]*|разм\.?|р-р|р\.?|size)"
# «размер L», «р.XL», «размер (межд.) XXL», «размер одежды: M»
KEYWORD_BEFORE = re.compile(rf"(?<![^\W\d_]){SIZE_WORD}(?:\s+одежды|\s+междунар[а-я.]*|\s*\([^)]{{0,15}}\))?"
                            r"\s*[:\-]?\s*$", re.IGNORECASE)
RU_MIN, RU_MAX = 38, 80  # Russian adult clothing sizes in KTRU
GIRTH_MIN = 76  # chest girth of the smallest adult size (38)
CHILD_WORDS = {"детский", "девочка", "мальчик", "ребенок", "подросток", "подростковый"}
# «р. 52-54», «размер 48», «р. 96-100/182-188», «104/108» (chest girth [/ height]); not «размер 56 см» (a head size)
RU_SIZE_RE = re.compile(rf"(?<![^\W\d_])({SIZE_WORD})\s*[:\-]?\s*(\d{{2,3}})(?:\s*([-/])\s*(\d{{2,3}}))?"
                        r"(?:\s*/\s*(\d{2,3}(?:\s*-\s*\d{2,3})?))?(?![\w.,]?\d)(?!\s*(?:см|мм|cm)(?![а-яa-z]))",
                        re.IGNORECASE)


def canonical_letter(raw: str) -> str | None:
    """«xl», «ХL», «2XL», «XXXXL» → canonical letter size or None."""
    s = raw.translate(LOOKALIKE).upper()
    m = re.fullmatch(r"([2-7]?)(X*)([SML])", s)
    if not m:
        return None
    digit, xs, letter = m.groups()
    if letter == "M" and (xs or digit):
        return None  # «XM» is not a size
    count = int(digit) - 1 + len(xs) if digit else len(xs)
    if digit and not xs:
        return None
    if letter == "S":
        return {0: "S", 1: "XS", 2: "XXS"}.get(count)
    if letter == "M":
        return "M"
    return {0: "L", 1: "XL", 2: "XXL", 3: "XXXL"}.get(count, f"{count}XL" if count <= 7 else None)


def _letter_range(first: str, second: str | None) -> list[str]:
    """«S-M» → [S, M]; «L/XL» → [L, XL]; a single letter → [it]."""
    if not second:
        return [first]
    i, j = sorted((SIZE_ORDER.index(first), SIZE_ORDER.index(second)))
    return SIZE_ORDER[i:j + 1]


def _ru_value(v: dict) -> tuple[int, int] | None:
    """A KTRU value of a Russian size characteristic → (low, high): 52 → (52, 52), «52-54» → (52, 54)."""
    if "text" in v:
        m = re.fullmatch(r"\s*(\d{2})(?:\s*-\s*(\d{2}))?\s*", v["text"])
        return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None
    if v.get("min") is not None and v.get("min") == v.get("max") and float(v["min"]).is_integer():
        return int(v["min"]), int(v["min"])
    return None


def is_letter_size_char(ch: dict) -> bool:
    texts = [v.get("text") for v in ch["values"]]
    return bool(texts) and all(t and canonical_letter(t.strip()) for t in texts)


def is_ru_size_char(ch: dict) -> bool:
    """«Российский размер» and «Размер» with clothing sizes 38…80 (not head sizes 54…63 in centimetres)."""
    if ch["name"] not in RU_SIZE_NAMES or ch.get("unit"):
        return False
    ranges = [_ru_value(v) for v in ch["values"]]
    return len(ranges) >= 4 and all(r and 38 <= r[0] <= r[1] <= 80 and r[0] % 2 == 0 for r in ranges)


class SizeMatcher:
    def __init__(self, store: KtruStore):
        self.store = store

    def _size_chars(self, codes: list[str]) -> dict[str, dict]:
        """name → {kind: letter | ru, by_code: {code: [values]}}."""
        chars: dict[str, dict] = {}
        for code in codes:
            for ch in self.store.by_code[code]["characteristics"]:
                kind = "letter" if is_letter_size_char(ch) else "ru" if is_ru_size_char(ch) else None
                if kind:
                    chars.setdefault(ch["name"], {"kind": kind, "by_code": {}})["by_code"][code] = ch["values"]
        return chars

    @staticmethod
    def _letter_mentions(text: str, clothing: bool) -> list[dict]:
        out = []
        for m in LETTER_RE.finditer(text):
            first, second = canonical_letter(m.group(1)), canonical_letter(m.group(2)) if m.group(2) else None
            if not first or (m.group(2) and not second):
                continue
            before = text[:m.start()]
            keyword = KEYWORD_BEFORE.search(before)
            prev = before.rstrip()[-1:]
            ru = RU_AFTER_LETTER.match(text, m.end())
            ru_range = (int(ru.group(1)), int(ru.group(2) or ru.group(1))) if ru else None
            if ru_range and not (38 <= ru_range[0] <= ru_range[1] <= 80 and ru_range[0] % 2 == 0):
                ru_range = None  # «L/17» — a collar or a height group, not a Russian size
            nxt = text[m.end():].lstrip()[:1]
            fragment = re.fullmatch(r"\s*", re.split(r"[,;\n]", before)[-1]) and \
                re.fullmatch(r"\s*", re.split(r"[,;\n]", text[m.end():])[0])
            raw = m.group()
            # without a size word: written in capitals («футболка XL», «М»); not «2 М» (metres) or «M 12» (a model),
            # but «48 L», «XL 54-56» are sizes; a lowercase letter only as a whole fragment («куртка, xl»)
            if not keyword:
                metres = first == "M" and not second
                if not clothing or (prev.isdigit() and metres) or (nxt.isdigit() and not ru_range):
                    continue
                if raw != raw.upper() and not fragment:
                    continue
            out.append({"start": keyword.start() if keyword else m.start(), "end": ru.end() if ru_range else m.end(),
                        "value_start": m.start(), "letters": _letter_range(first, second), "raw": raw,
                        "keyword": bool(keyword), "ru_range": ru_range})
        return out

    def match(self, text: str, codes: list[str]) -> tuple[list[dict], list[dict], list[list[int]], list[dict]]:
        """→ (characteristics, errors, reserved spans, sizes without a KTRU characteristic for the portal / new level)."""
        chars = self._size_chars(codes)
        rank = {c: i for i, c in enumerate(codes)}
        clothing = bool(chars) or any(self.store.by_code[c]["category"] == "одежда" for c in codes[:1])
        lemmas = {t["lemma"] for t in tokenize(text)}
        table = "women" if "женский" in lemmas and "мужской" not in lemmas else "men"
        children = bool(lemmas & CHILD_WORDS)
        found, errors, reserved, unmatched = [], [], [], []
        tokens = tokenize(text)

        def reserve(start, end):
            reserved.extend(t["span"] for t in tokens if start <= t["span"][0] and t["span"][1] <= end)

        # «р. 52-54», «размер 48», «р. 96-100/182-188»: Russian sizes — only when the candidates have such
        # a characteristic. Workwear is often sized by chest girth / height: «96-100» cm = Russian size 48–50
        # (size = half the chest girth). Other numbers («р. 26», «р. 146» — children's, «р. 45») are left to the
        # other matchers.
        ru = {n: info for n, info in chars.items() if info["kind"] == "ru"}
        for m in RU_SIZE_RE.finditer(text):
            if not ru or any(s < m.end() and m.start() < e for s, e in reserved):
                continue
            lo, second, height = int(m.group(2)), int(m.group(4)) if m.group(4) else None, m.group(5)
            if second is not None and m.group(3) == "/" and not 0 < second - lo <= 8:
                second, height = None, height or m.group(4)  # «96/182-188» — girth / height, not a range
            hi = second or lo
            source = text[m.start(2):m.end(4) if second else m.end(2)]
            span = [m.start(2), m.start(2) + len(source)]
            girth = lo > RU_MAX or (height and lo >= GIRTH_MIN)  # «р. 80/170-176» is a girth, «р. 80» a size
            if not girth and RU_MIN <= lo <= hi <= RU_MAX and lo % 2 == 0:
                size_lo, size_hi, method = lo, hi, "number"
            elif girth and not children and GIRTH_MIN <= lo <= hi <= 2 * RU_MAX:
                # chest girth c fits size c/2 ± 1 (a size step is 4 cm of girth): 90 → 44–46, 96-100 → 48–50
                size_lo, size_hi, method = lo / 2 - 1, hi / 2 + 1, "size_table"
            else:
                continue
            hits = []
            for name, info in ru.items():
                codes_ok, exact, sizes = [], None, set()
                for c, values in info["by_code"].items():
                    # «52» fits 52, «50-52» and «52-54»; «52-54» fits 52, 54 and «52-54»
                    fits = [v for v in values if (r := _ru_value(v)) and (
                        size_lo <= r[0] and r[1] <= size_hi or (size_lo == size_hi and r[0] <= size_lo <= r[1]))]
                    if fits:
                        codes_ok.append(c)
                        exact = exact or next((format_value(v) for v in fits
                                               if _ru_value(v) == (size_lo, size_hi)), None)
                        sizes |= {x for v in fits for x in _ru_value(v)}
                if codes_ok:
                    low, high = (size_lo, size_hi) if method == "number" else (min(sizes), max(sizes))
                    value = exact or (f"{low}" if low == high else f"{low}–{high}")
                    hits.append(self._entry(name, value, source, span, method, codes_ok, rank, source))
            if not hits and method != "number":
                continue  # a girth that fits no size may be something else — not an error of ours
            hits = self._one_per_kind(hits, chars, rank)
            found += hits
            if not hits:  # «р. 72» when KTRU has sizes up to 70
                name, info = next(iter(ru.items()))
                errors.append(self._error(name, source, span, f"Значение «{source}» недопустимо", self._allowed(info)))
            reserve(m.start(), span[1])
        # letters after numbers: «размер 48 (M)» keeps the Russian size the user wrote
        for m in self._letter_mentions(text, clothing):
            if any(s < m["end"] and m["value_start"] < e for s, e in reserved):
                continue
            source = text[m["value_start"]:m["end"]]
            span = [m["value_start"], m["end"]]
            hits, errs = [], []
            for name, info in chars.items():
                if info["kind"] == "letter":
                    codes_ok = [c for c, values in info["by_code"].items()
                                if any(canonical_letter(v["text"].strip()) in m["letters"] for v in values)]
                    if codes_ok:
                        # the value as written in KTRU (it may be the cyrillic «М»)
                        value = next(v["text"] for vals in info["by_code"].values() for v in vals
                                     if canonical_letter(v["text"].strip()) in m["letters"])
                        value = value if len(m["letters"]) == 1 else f"{m['letters'][0]}–{m['letters'][-1]}"
                        hits.append(self._entry(name, value, source, span, "exact", codes_ok, rank, m["raw"]))
                    else:
                        errs.append(self._error(name, source, span, f"Размер «{source}» недопустим", self._allowed(info)))
                elif not children or m["ru_range"]:
                    # the Russian size written by the user («L/48-50») wins over the table
                    lo, hi = m["ru_range"] or (min(SIZE_TABLE[table][x][0] for x in m["letters"]),
                                               max(SIZE_TABLE[table][x][1] for x in m["letters"]))
                    codes_ok = [c for c, values in info["by_code"].items()
                                if any((r := _ru_value(v)) and lo <= r[0] and r[1] <= hi for v in values)]
                    if codes_ok:
                        hits.append(self._entry(name, f"{lo}" if lo == hi else f"{lo}–{hi}", source, span,
                                                "size_table", codes_ok, rank, m["raw"]))
                    else:
                        errs.append(self._error(name, source, span,
                                                f"Размер «{source}» (≈ {lo}–{hi} по российской шкале) недопустим",
                                                self._allowed(info)))
            hits = self._one_per_kind(hits, chars, rank)
            found += hits
            errors += [] if hits else errs
            reserve(m["start"], m["end"])
            if not hits and not errs:
                unmatched.append({"source": source, "span": span, "a": None, "b": None, "unit_raw": None,
                                  "value_text": source.upper().translate(LOOKALIKE).upper(), "dimension": None,
                                  "number_spans": [], "synthetic": True, "keyword": SIZE_NAME,
                                  "keyword_span": None})

        uniq = {}
        for f in found:
            uniq.setdefault(f["name"], f)
        return list(uniq.values()), errors, reserved, unmatched

    @staticmethod
    def _one_per_kind(hits: list[dict], chars: dict, rank: dict) -> list[dict]:
        """One size is one characteristic of each kind: «Российский размер» rather than also «Размер» of a few
        candidates — the one that fits more codes, then the better-ranked one."""
        best: dict[str, dict] = {}
        for h in hits:
            kind = chars[h["name"]]["kind"]
            key = (-len(h["codes"]), min(rank[c] for c in h["codes"]))
            if kind not in best or key < (-len(best[kind]["codes"]), min(rank[c] for c in best[kind]["codes"])):
                best[kind] = h
        return list(best.values())

    @staticmethod
    def _entry(name, value, source, span, method, codes_ok, rank, user_value) -> dict:
        return {"name": name, "value": value, "unit": None, "user_value": user_value, "source": source, "span": span,
                "method": method, "score": 95, "codes": sorted(codes_ok, key=lambda c: rank[c]),
                "value_text": user_value}

    @staticmethod
    def _allowed(info: dict) -> str:
        values = {format_value(v) for vals in info["by_code"].values() for v in vals}
        return ", ".join(sorted(values, key=lambda s: (len(s), s)))

    @staticmethod
    def _error(name, source, span, what, allowed) -> dict:
        return {"type": "nonexistent_value", "characteristic": name, "value": source, "source": source, "span": span,
                "message": f"{what} для характеристики «{name}». Допустимые значения: {allowed}."}
