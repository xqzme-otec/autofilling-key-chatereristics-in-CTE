from __future__ import annotations

"""Исправление опечаток в сторону терминов КТРУ.

Проверяем каждое слово:
  1. есть в словаре КТРУ (как есть или по лемме) — ок;
  2. обычное русское слово (pymorphy его знает) — не трогаем;
  3. иначе пробуем: неправильная раскладка (cnek → стул), затем SymSpell —
     ближайшее слово КТРУ с расстоянием Дамерау–Левенштейна ≤ 2
     (замена, вставка, удаление, перестановка соседних букв).
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
# латинские буквы + клавиши, на которых в русской раскладке стоят буквы (ж, э, х, ъ, б, ю, ё)
LAYOUT_RE = re.compile(r"[A-Za-z\[\];',.`]*[A-Za-z][A-Za-z\[\];',.`]*")


class Corrector:
    def __init__(self, store: KtruStore):
        # синонимы («кожзам», «хб») — тоже правильные слова, хоть их и нет в КТРУ
        # synonyms («кожзам», «хб») and unit words («мкг», «квт») are valid words too, though not in KTRU
        self.vocab = store.vocab + Counter(load_synonyms().keys()) + Counter(USER_UNITS.keys()) + Counter(UNIT_WORDS)
        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # в словарь исправлений — все формы слов КТРУ, чтобы «тубмой» исправлялось в «тумбой»
        for word, freq in self.vocab.items():
            for form in word_forms(word) | {word}:
                self.sym.create_dictionary_entry(form, freq)
        # в каких позициях КТРУ встречается слово — чтобы выбирать исправление по контексту
        self.postings: dict[str, set] = {}
        for i, item in enumerate(store.items):
            for t in tokenize(store.passage(item)):
                if t["text"][0].isalpha():
                    self.postings.setdefault(t["lemma"], set()).add(i)

    def _in_vocab(self, word: str) -> bool:
        w = word.lower()
        return w in self.vocab or lemma(w) in self.vocab

    def _context_score(self, term: str, context: list[str]) -> int:
        """Сколько раз вариант встречается в одних позициях КТРУ с остальными словами запроса."""
        items = self.postings.get(lemma(term), set())
        return sum(len(items & self.postings.get(w, set())) for w in context)

    def _fix(self, word: str, known: bool, context: list[str] = ()) -> tuple[str, str, list[str]] | None:
        """Возвращает (исправление, причина, другие варианты) или None, если слово в порядке.

        Из равноудалённых вариантов («стл» → «стол» / «стул») выбираем тот, что чаще
        встречается в КТРУ вместе с остальными словами запроса («на колесиках», «кожзам»).
        """
        if self._in_vocab(word):
            return None

        if known or all(c.isascii() for c in word):
            return None  # латиница после fix_layout — бренды, модели, марки: не трогаем

        max_dist = 1 if len(word) <= 4 else 2
        # сначала само слово, затем его лемма: в словаре КТРУ «женский», а не «жеенская»
        for w in dict.fromkeys([word.lower(), lemma(word)]):
            found = self.sym.lookup(w, Verbosity.CLOSEST, max_edit_distance=max_dist)
            if found:
                # разные формы одного слова («стола», «столом») — один вариант
                by_lemma = {}
                for sug in found:
                    by_lemma.setdefault(lemma(sug.term), sug)
                ranked = sorted(by_lemma.values(), key=lambda s: (self._context_score(s.term, context), s.count),
                                reverse=True)
                best = ranked[0]
                # в варианты — только слова, которые реально встречаются в позициях КТРУ («ст», «стал» — нет)
                alts = [s.term for s in ranked[1:] if len(s.term) > 2 and lemma(s.term) in self.postings
                        and pos(s.term) == pos(best.term)][:2]
                return best.term, f"опечатка (отличается на {best.distance} симв.)", alts
        return None

    def fix_layout(self, text: str) -> tuple[str, list[dict]]:
        """Слова, набранные в английской раскладке: «rjv,bybhjdfyysq» → «комбинированный».

        Замена посимвольная, длина текста не меняется — спаны токенов остаются верными.
        """
        corrections, chars = [], list(text)
        for m in LAYOUT_RE.finditer(text):
            chunk = m.group().rstrip(",.")  # запятая/точка в конце — скорее всего пунктуация
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
        """Возвращает исправленный текст и список исправлений для показа пользователю."""
        corrections, out, pos = [], [], 0
        for t in tokens:
            # the order of dimensions «ШхГхВ» is not a typo («шагов»)
            if not t["text"][0].isalpha() or len(t["text"]) < 3 or AXES_HEADER_RE.fullmatch(t["text"]):
                continue
            context = [o["lemma"] for o in tokens if o is not t and o["text"][0].isalpha() and self._in_vocab(o["text"])]
            fix = self._fix(t["text"], t.get("known", False), context)
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
