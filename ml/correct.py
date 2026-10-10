from __future__ import annotations

"""Typo correction towards KTRU terms.

Each word is checked:
  1. it is in the KTRU dictionary (as is or by lemma) — fine;
  2. it is an ordinary Russian word (known to pymorphy) — left alone;
  3. otherwise we try the wrong keyboard layout (cnek → стул), then SymSpell —
     the nearest KTRU word within Damerau–Levenshtein distance ≤ 2
     (substitution, insertion, deletion, transposition of adjacent letters).
"""
import re
from collections import Counter

from symspellpy import SymSpell, Verbosity

from ml.ktru import KtruStore
from ml.text import is_known, lemma, pos, tokenize, word_forms
from ml.units import AXES_HEADER_RE, UNIT_WORDS, USER_UNITS
from ml.values import load_synonyms

EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
RU = "йцукенгшщзхъфывапролджэячсмитьбюё"
EN_TO_RU = str.maketrans(EN + EN.upper(), RU + RU.upper())
# latin letters + keys that carry letters in the Russian layout (ж, э, х, ъ, б, ю, ё)
LAYOUT_RE = re.compile(r"[A-Za-z\[\];',.`]*[A-Za-z][A-Za-z\[\];',.`]*")


class Corrector:
    def __init__(self, store: KtruStore):
        # synonyms («кожзам», «хб») and unit words («мкг», «квт») are valid words too, though not in KTRU
        self.vocab = store.vocab + Counter(load_synonyms().keys()) + Counter(USER_UNITS.keys()) + Counter(UNIT_WORDS)
        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # all forms of KTRU words go to the correction dictionary, so that «тубмой» is corrected to «тумбой»
        for word, freq in self.vocab.items():
            for form in word_forms(word) | {word}:
                self.sym.create_dictionary_entry(form, freq)
        # which KTRU positions contain a word — to pick a correction by context
        self.postings: dict[str, set] = {}
        for i, item in enumerate(store.items):
            for t in tokenize(store.passage(item)):
                if t["text"][0].isalpha():
                    self.postings.setdefault(t["lemma"], set()).add(i)
        self.products = Counter()
        for item in store.items:
            noun = next((t["lemma"] for t in tokenize(item["name"]) if pos(t["text"]) == "NOUN"), None)
            if noun:
                self.products[noun] += 1

    def _in_vocab(self, word: str) -> bool:
        w = word.lower()
        return w in self.vocab or lemma(w) in self.vocab

    def _context_score(self, term: str, context: list[str]) -> int:
        """How often a variant occurs in the same KTRU positions as the other words of the request."""
        items = self.postings.get(lemma(term), set())
        return sum(len(items & self.postings.get(w, set())) for w in context)

    def _fix(self, word: str, known: bool, context: list[str] = (), product: bool = False) -> tuple[str, str, list[str]] | None:
        """Returns (correction, reason, other variants) or None if the word is fine.

        Among equally distant variants («стл» → «стол» / «стул») the one that occurs in KTRU
        together with the other words of the request («на колесиках», «кожзам») more often wins.
        """
        if self._in_vocab(word):
            return None

        if product and len(word) >= 4 and not all(c.isascii() for c in word):
            from rapidfuzz.distance import DamerauLevenshtein
            options = [(self._context_score(noun, context), count, noun)
                       for noun, count in self.products.items()
                       if DamerauLevenshtein.distance(word.lower(), noun) <= 1]
            if options:
                _, _, noun = max(options)
                if noun != lemma(word) and (not known or self._context_score(noun, context) > 0
                                           or len(options) == 1):
                    return noun, "уточнение названия товара по КТРУ", []

        if known or all(c.isascii() for c in word):
            return None  # latin words left after fix_layout are brands, models, grades: left alone

        max_dist = 1 if len(word) <= 4 else 2
        # the word itself first, then its lemma: the KTRU dictionary has «женский», not «жеенская»
        for w in dict.fromkeys([word.lower(), lemma(word)]):
            found = self.sym.lookup(w, Verbosity.CLOSEST, max_edit_distance=max_dist)
            if found:
                # different forms of one word («стола», «столом») are one variant
                by_lemma = {}
                for sug in found:
                    by_lemma.setdefault(lemma(sug.term), sug)
                ranked = sorted(by_lemma.values(), key=lambda s: (self._context_score(s.term, context), s.count),
                                reverse=True)
                best = ranked[0]
                # alternatives are only words that really occur in KTRU positions (not «ст», «стал»)
                alts = [s.term for s in ranked[1:] if len(s.term) > 2 and lemma(s.term) in self.postings
                        and pos(s.term) == pos(best.term)][:2]
                return best.term, f"опечатка (отличается на {best.distance} симв.)", alts
        return None

    def fix_layout(self, text: str) -> tuple[str, list[dict]]:
        """Words typed in the English layout: «rjv,bybhjdfyysq» → «комбинированный».

        The replacement is character-by-character, the text length does not change — token spans stay valid.
        """
        corrections, chars = [], list(text)
        for m in LAYOUT_RE.finditer(text):
            chunk = m.group().rstrip(",.")  # a trailing comma/dot is most likely punctuation
            if len(chunk) < 3:
                continue
            switched = chunk.translate(EN_TO_RU).lower()
            words = re.findall(r"[а-яё]+", switched)
            if words and all(self._in_vocab(w) or is_known(w) for w in words):
                start, end = m.start(), m.start() + len(chunk)
                chars[start:end] = switched
                corrections.append({"from": chunk, "to": switched, "span": [start, end],
                                    "reason": "неправильная раскладка клавиатуры"})
        return "".join(chars), corrections

    def correct(self, text: str, tokens: list[dict]) -> tuple[str, list[dict]]:
        """Returns the corrected text and the list of corrections to show to the user."""
        corrections, out, pos = [], [], 0
        first_word = next((t for t in tokens if t["text"][0].isalpha()), None)
        for t in tokens:
            # the order of dimensions «ШхГхВ» is not a typo («шагов»)
            if not t["text"][0].isalpha() or len(t["text"]) < 3 or AXES_HEADER_RE.fullmatch(t["text"]):
                continue
            context = [o["lemma"] for o in tokens if o is not t and o["text"][0].isalpha() and self._in_vocab(o["text"])]
            fix = self._fix(t["text"], t.get("known", False), context, t is first_word)
            if fix is None:
                continue
            new, reason, alternatives = fix
            start, end = t["span"]
            out += [text[pos:start], new]
            pos = end
            corr = {"from": t["text"], "to": new, "span": t["span"], "reason": reason}
            if alternatives:
                corr["alternatives"] = alternatives
            corrections.append(corr)
        out.append(text[pos:])
        return "".join(out), corrections
