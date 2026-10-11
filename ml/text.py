from __future__ import annotations

"""Tokens and lemmas.

In production the backend sends tokens with lemmas (the `tokens` field of the input JSON).
This is a temporary pymorphy3 replacement so that the ML part runs and is tested
without the backend. The token format is the same as in the backend contract.
"""
import re
from functools import lru_cache

import pymorphy3

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*|\d+(?:[.,]\d+)?|[^\sA-Za-zА-Яа-яЁё\d]")

_morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=100_000)
def lemma(word: str) -> str:
    return _morph.parse(word.lower())[0].normal_form.replace("ё", "е")


@lru_cache(maxsize=100_000)
def is_known(word: str) -> bool:
    return _morph.word_is_known(word.lower())


_NORMALIZE = str.maketrans({"ё": "е", "Ё": "Е", "«": '"', "»": '"', "„": '"', "“": '"', "”": '"',
                            "×": "x", " ": " ", "\t": " ", "—": "-", "–": "-"})


def normalize(text: str) -> str:
    """Character-by-character normalization: the text length does not change, spans stay valid."""
    text = text.translate(_NORMALIZE)
    return re.sub(r"(?i)\bх/б\b", lambda m: m.group()[0] + "б ", text)  # «х/б» → «хб »


def tokenize(text: str) -> list[dict]:
    tokens = []
    for m in TOKEN_RE.finditer(text):
        t = m.group()
        is_word = t[0].isalpha()
        tokens.append({
            "text": t,
            "lemma": lemma(t) if is_word else t.lower(),
            "span": [m.start(), m.end()],
            "known": is_known(t) if is_word else True,
        })
    return tokens


def lemmatize_phrase(text: str) -> str:
    return " ".join(t["lemma"] for t in tokenize(text) if t["text"][0].isalnum())


def word_forms(word: str) -> set[str]:
    """All forms of a word: «тумба» → {тумба, тумбы, тумбой, ...}. Empty for unknown words."""
    if not is_known(word):
        return set()
    return {f.word.replace("ё", "е") for f in _morph.parse(word.lower())[0].lexeme}


def main_noun(text: str) -> str | None:
    """Lemma of the first noun — usually the product itself («офисный стул» → «стул»)."""
    for t in tokenize(text):
        if t["text"][0].isalpha() and len(t["text"]) > 2:
            p = _morph.parse(t["text"].lower())[0]
            if p.tag.POS == "NOUN":
                return p.normal_form.replace("ё", "е")
    return None


def pos(word: str) -> str | None:
    """Part of speech (NOUN, ADJF, VERB, …) of the most probable parse."""
    return _morph.parse(word.lower())[0].tag.POS


def to_nominative(words: list[str]) -> str:
    """«противоскользящей рукояткой» → «противоскользящая рукоятка»: nouns to nominative, adjectives agree."""
    parses = [_morph.parse(w.lower())[0] for w in words]
    head = next((p for p in reversed(parses) if p.tag.POS == "NOUN"), None)
    gender = head.tag.gender if head else None
    number = head.tag.number if head else "sing"
    out = []
    for w, p in zip(words, parses):
        if p.tag.POS == "NOUN" and p is head:
            form = p.inflect({"nomn", number}) if number else p.inflect({"nomn"})
        elif p.tag.POS in ("ADJF", "PRTF"):
            grammemes = {"nomn", "plur"} if number == "plur" else {"nomn", gender} if gender else {"nomn"}
            form = p.inflect(grammemes)
        else:
            form = None
        out.append(form.word if form else w.lower())
    return " ".join(out)
