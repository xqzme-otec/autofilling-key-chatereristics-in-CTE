from __future__ import annotations

"""Токены и леммы.

В продакшене токены с леммами присылает бэкенд (поле `tokens` во входном JSON).
Здесь — временная замена на pymorphy3, чтобы ML-часть работала и тестировалась
без бэкенда. Формат токена такой же, как в контракте с бэкендом.
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
    """Посимвольная нормализация: длина текста не меняется, спаны остаются верными."""
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
    """Все словоформы слова: «тумба» → {тумба, тумбы, тумбой, ...}. Для незнакомых слов — пусто."""
    if not is_known(word):
        return set()
    return {f.word.replace("ё", "е") for f in _morph.parse(word.lower())[0].lexeme}


def main_noun(text: str) -> str | None:
    """Лемма первого существительного — обычно это сам товар («офисный стул» → «стул»)."""
    for t in tokenize(text):
        if t["text"][0].isalpha() and len(t["text"]) > 2:
            p = _morph.parse(t["text"].lower())[0]
            if p.tag.POS == "NOUN":
                return p.normal_form.replace("ё", "е")
    return None


def pos(word: str) -> str | None:
    """Часть речи (NOUN, ADJF, VERB, …) самого вероятного разбора."""
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
