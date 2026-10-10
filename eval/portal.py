"""Проверка на реальных карточках товаров (СТЕ) Портала поставщиков.

    .venv/bin/python -m eval.portal make     # data/raw/supplier_portal/*.xlsx → eval/cases_portal.jsonl
    .venv/bin/python -m eval.portal run [-v] # отчёт

В выгрузках нет кода КТРУ, поэтому эталон грубый — категория файла (Инструменты/Мебель/Одежда).
Запрос — название СТЕ (как пишут поставщики: бренды, модели, артикулы), во втором варианте —
название + 2 характеристики из карточки. Метрики:
  found        — товар найден в КТРУ (не «product_not_found»);
  category_ok  — категория лучшего кандидата совпала с категорией файла (среди найденных);
  narrowed     — дошли до кода или до уточняющего вопроса;
  chars_avg    — сколько характеристик извлечено в среднем (КТРУ, Портал и новые);
  errors_per_100 — ошибки (severity=error) на 100 запросов, кроме «не найден»: карточки реальные,
                   поэтому это в основном ложные срабатывания или расхождения портала с КТРУ;
  warnings_per_100 — предупреждения о подозрительных значениях.
"""
import argparse
import json
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "supplier_portal"
CASES = Path(__file__).resolve().parent / "cases_portal.jsonl"
CATEGORIES = {"Инструменты": "инструменты", "Мебель": "мебель", "Одежда": "одежда"}
JUNK_VALUE = re.compile(r"^(\(не задано\)|0(\.0+)?)(\s|$)")


def parse_characteristics(raw: str) -> list[str]:
    """«Вес 2.03000 кг | Мощность 10.80000 Вт» → ["Вес 2.03 кг", "Мощность 10.8 Вт"] без пустых значений."""
    out = []
    for part in str(raw or "").split("|"):
        part = re.sub(r"(\d+\.\d*?)0+(\s|$)", r"\1\2", part.strip()).replace(". ", " ").strip()
        part = re.sub(r"(\d)\.(\s|$)", r"\1\2", part)
        name_value = part.split(" ", 1)
        if len(name_value) == 2 and not JUNK_VALUE.match(name_value[1]) and " 0 " not in f" {name_value[1]} ":
            out.append(part)
    return out


def make(per_category: int = 200, rows: int = 30000, seed: int = 0):
    import pandas as pd

    rng = random.Random(seed)
    cases = []
    for file, category in CATEGORIES.items():
        path = RAW_DIR / f"{file}.xlsx"
        df = pd.read_excel(path, sheet_name=0, nrows=rows, usecols=["ID СТЕ", "Наименование СТЕ", "Характеристики"])
        df = df.dropna(subset=["Наименование СТЕ"]).drop_duplicates("Наименование СТЕ")
        for r in df.sample(min(per_category, len(df)), random_state=rng.randrange(10**6)).itertuples(index=False):
            name = str(r[1]).strip()
            chars = parse_characteristics(r[2])
            cases.append({"id": int(r[0]), "category": category, "text": name, "variant": "name"})
            if chars:
                extra = ", ".join(rng.sample(chars, min(2, len(chars))))
                cases.append({"id": int(r[0]), "category": category, "text": f"{name}, {extra}", "variant": "name+chars"})
        print(f"{file}: прочитано {len(df)} карточек")
    with CASES.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cases)} запросов → {CASES}")


def run(verbose: bool = False):
    from ml.pipeline import Pipeline

    cases = [json.loads(l) for l in CASES.open(encoding="utf-8")]
    pipe = Pipeline()
    pipe.analyze("прогрев")
    groups: dict[tuple, list] = {}
    error_types = Counter()
    origins = Counter()
    for c in cases:
        t = time.perf_counter()
        r = pipe.analyze(c["text"])
        ms = (time.perf_counter() - t) * 1000
        found = not any(e["type"] == "product_not_found" for e in r["errors"])
        top_cat = pipe.store.by_code[r["candidates"][0]["code"]]["category"] if r["candidates"] else None
        errs = [e for e in r["errors"] if e["severity"] == "error" and e["type"] != "product_not_found"]
        warns = [e for e in r["errors"] if e["severity"] == "warning"]
        error_types.update(e["type"] for e in errs + warns)
        origins.update(ch["origin"] for ch in r["characteristics"])
        groups.setdefault((c["category"], c["variant"]), []).append({
            "found": found, "cat_ok": found and top_cat == c["category"],
            "narrowed": bool(r["ktru"] or r["next_step"]), "chars": len(r["characteristics"]),
            "errors": len(errs), "warnings": len(warns), "ms": ms})
        if verbose and (errs or (found and top_cat != c["category"])):
            print(f"  ✗ [{c['category']}] {c['text'][:110]!r}\n    лучший кандидат: "
                  f"{r['candidates'][0]['name'] if r['candidates'] else '—'} ({top_cat}); "
                  f"ошибки: {[e['message'][:90] for e in errs]}")

    def report(rows):
        n = len(rows)
        found = [x for x in rows if x["found"]]
        return {"n": n, "found": round(len(found) / n, 3),
                "category_ok": round(sum(x["cat_ok"] for x in found) / max(len(found), 1), 3),
                "narrowed": round(sum(x["narrowed"] for x in rows) / n, 3),
                "chars_avg": round(statistics.mean(x["chars"] for x in rows), 2),
                "errors_per_100": round(100 * sum(x["errors"] for x in rows) / n, 1),
                "warnings_per_100": round(100 * sum(x["warnings"] for x in rows) / n, 1),
                "ms_mean": round(statistics.mean(x["ms"] for x in rows), 1)}

    for key in sorted(groups):
        print(f"{key[0]:12} {key[1]:11}", report(groups[key]))
    print(f"{'ВСЕГО':24}", report([x for rows in groups.values() for x in rows]))
    print("типы ошибок и предупреждений:", dict(error_types))
    print("происхождение характеристик:", dict(origins))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    make() if args.cmd == "make" else run(args.verbose)
