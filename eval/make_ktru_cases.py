"""Тестовые запросы из настоящего КТРУ: название (иногда сокращённое) + 1–2 значения характеристик.

    .venv/bin/python -m eval.make_ktru_cases            # → eval/cases_ktru.jsonl

Ожидаемый ответ: код позиции, её название и указанные характеристики. Это приближение
к реальным запросам — на xlsx-выгрузках цифры будут честнее.
"""
import json
import random
from collections import Counter

from ml.ktru import KtruStore, is_bool_char, is_text_char

OUT = "eval/cases_ktru.jsonl"


def main(seed: int = 0, per_name: int = 1):
    rng = random.Random(seed)
    store = KtruStore()
    by_name: dict[str, list] = {}
    for it in store.items:
        by_name.setdefault(it["name"], []).append(it)

    cases = []
    for name, items in by_name.items():
        for it in rng.sample(items, min(per_name, len(items))):
            # значения, которые однозначно указывают на характеристику внутри позиции
            counts = Counter(v["text"].lower() for ch in it["characteristics"] if is_text_char(ch) for v in ch["values"])
            opts = [(ch["name"], v["text"]) for ch in it["characteristics"]
                    if is_text_char(ch) and not is_bool_char(ch) and len(ch["values"]) > 1
                    for v in ch["values"] if counts[v["text"].lower()] == 1 and len(v["text"]) < 40]
            rng.shuffle(opts)
            picked, used_chars = [], set()
            for ch_name, v in opts:  # не больше одного значения на характеристику
                if ch_name not in used_chars and len(picked) < rng.choice([1, 2]):
                    picked.append((ch_name, v))
                    used_chars.add(ch_name)
            title = name.lower()
            if len(title.split()) > 3 and rng.random() < 0.5:  # сокращённое название: первые 2 значимых слова
                kept, content = [], 0
                for w in title.split():
                    kept.append(w)
                    content += len(w) > 3
                    if content == 2:
                        break
                short = " ".join(kept)
                # сокращаем, только если сокращение не подходит к другим названиям
                if sum(n.lower().startswith(short) for n in by_name) == 1:
                    title = short
            text = ", ".join([title] + [v.lower() for _, v in picked])
            cases.append({"text": text, "ktru": it["code"], "name": name, "chars": dict(picked)})

    with open(OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} примеров → {OUT}")


if __name__ == "__main__":
    main()
