"""Text corruption for robustness checks: typos, swaps, dropped letters, keyboard layout."""
import random

RU = "йцукенгшщзхъфывапролджэячсмитьбю"
EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,."
RU_TO_EN = str.maketrans(RU, EN)


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["swap", "drop", "replace", "double"])
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "replace":
        return word[:i] + rng.choice(RU) + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def add_noise(text: str, rng: random.Random, p_typo: float = 0.3, p_layout: float = 0.05) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        r = rng.random()
        if r < p_layout and w.isalpha():
            out.append(w.lower().translate(RU_TO_EN))
        elif r < p_layout + p_typo and w.isalpha():
            out.append(_typo(w, rng))
        else:
            out.append(w)
    return " ".join(out)
