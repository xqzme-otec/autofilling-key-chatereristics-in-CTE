"""Russian number words → numbers: «полтора килограмма» → 1.5 кг, «два с половиной метра» → 2.5 м."""
from ml.text import lemma, tokenize

UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200,
    "триста": 300, "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900, "полтора": 1.5, "полторы": 1.5, "половина": 0.5, "четверть": 0.25,
}
THOUSAND = "тысяча"


def _value(word: str) -> float | None:
    w = word.lower().replace("ё", "е")
    if w in UNITS:
        return UNITS[w]
    lem = lemma(w)
    return UNITS.get(lem)


def find_number_words(text: str, is_unit) -> list[dict]:
    """Runs of number words followed by a unit word. `is_unit(word)` tells whether a word is a unit.

    Returns [{start, end, value, unit_raw}] with spans in `text`.
    """
    toks = [t for t in tokenize(text) if t["text"][0].isalpha()]
    out, i = [], 0
    while i < len(toks):
        word = toks[i]["text"].lower()
        # «полметра», «полкило» — one token
        if word.startswith("пол") and len(word) > 4 and is_unit(word[3:]):
            out.append({"start": toks[i]["span"][0], "end": toks[i]["span"][1], "value": 0.5, "unit_raw": word[3:]})
            i += 1
            continue
        if _value(word) is None and lemma(word) != THOUSAND:
            i += 1
            continue
        start, total, current, j = toks[i]["span"][0], 0.0, 0.0, i
        while j < len(toks):
            w = toks[j]["text"].lower()
            if lemma(w) == THOUSAND:
                current = (current or 1) * 1000
                total, current = total + current, 0.0
            elif (v := _value(w)) is not None:
                current += v
            else:
                break
            j += 1
        value = total + current
        # «два с половиной»
        if j + 1 < len(toks) and toks[j]["text"].lower() == "с" and lemma(toks[j + 1]["text"]) == "половина":
            value, j = value + 0.5, j + 2
        if j < len(toks) and is_unit(toks[j]["text"]):
            out.append({"start": start, "end": toks[j]["span"][1], "value": value, "unit_raw": toks[j]["text"]})
            i = j + 1
        else:
            i = j if j > i else i + 1
    return out
