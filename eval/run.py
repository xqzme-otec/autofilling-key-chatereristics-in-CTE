"""Оценка качества ML-части.

    python -m eval.run                         # чистые и зашумлённые тексты
    python -m eval.run --model deepvk/USER-bge-m3 --rerank --cases eval/cases.jsonl --ktru data/ktru.json

Метрики:
  name@1 / name@5 — правильное наименование товара первым / в первой пятёрке поиска;
  code_kept       — правильный код остался среди кандидатов после сужения (не потерян);
  false_not_found — доля запросов, где товар ошибочно объявлен «не найденным»;
  chars P/R/F1    — точность (значение допустимо для правильного кода) и полнота (по эталону);
  latency         — среднее и 95-й перцентиль времени на запрос.
"""
import argparse
import json
import random
import statistics
import time

from eval.noise import add_noise
from ml.ktru import MOCK_PATH, format_value
from ml.pipeline import Pipeline
from ml.search import EMB_MODEL
from ml.text import lemmatize_phrase


def evaluate(pipe: Pipeline, cases: list[dict], verbose: bool = False) -> dict:
    hit1 = hit5 = final = not_found = tp = tp_p = n_pred = n_gold = 0
    latencies = []
    for c in cases:
        want_name = c.get("name") or pipe.store.by_code[c["ktru"]]["name"]
        t = time.perf_counter()
        res = pipe.analyze(c["text"])
        latencies.append((time.perf_counter() - t) * 1000)

        # в КТРУ у одного названия много кодов — поиск оцениваем по названию, код выбирает сужение
        names = list(dict.fromkeys(x["name"] for x in pipe.searcher.search(res["text"], k=50)))
        hit1 += names[:1] == [want_name]
        hit5 += want_name in names[:5]
        final += c["ktru"] in [x["code"] for x in res["candidates"]] + [res["ktru"]]
        not_found += any(e["type"] == "product_not_found" for e in res["errors"])

        pred = {(x["name"], x["value"]) for x in res["characteristics"] if x["method"] != "name"}
        gold = set(c["chars"].items())
        # точность: значение допустимо для правильного кода (в эталоне не все верные характеристики —
        # например, «Тип каркаса = Деревянный» из названия «Стул на деревянном каркасе»)
        allowed = {ch["name"]: {lemmatize_phrase(format_value(v, ch["unit"])) for v in ch["values"]}
                   for ch in pipe.store.by_code[c["ktru"]]["characteristics"]}
        valid = {(n, v) for n, v in pred if lemmatize_phrase(v) in allowed.get(n, ())
                 or any(lemmatize_phrase(v) in a for a in allowed.get(n, ()))}
        tp_p += len(valid)
        tp += len(pred & gold)
        n_pred += len(pred)
        n_gold += len(gold)
        if verbose and (pred - valid or gold - pred or names[:1] != [want_name]):
            print(f"  ✗ {c['text']!r}\n    top1={names[:1]} want={want_name!r}\n"
                  f"    неверные={pred - valid} пропущены={gold - pred}")

    n = len(cases)
    p = tp_p / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {
        "name@1": round(hit1 / n, 3), "name@5": round(hit5 / n, 3), "code_kept": round(final / n, 3),
        "false_not_found": round(not_found / n, 3),
        "chars_P": round(p, 3), "chars_R": round(r, 3), "chars_F1": round(2 * p * r / (p + r) if p + r else 0, 3),
        "latency_ms_mean": round(statistics.mean(latencies), 1),
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (n - 1))], 1),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/cases_mock.jsonl")
    ap.add_argument("--ktru", default=None, help="путь к справочнику (по умолчанию: для *_mock — временные данные, иначе ktru.parquet)")
    ap.add_argument("--model", default=EMB_MODEL)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--noise-runs", type=int, default=3, help="сколько раз прогнать зашумлённую версию")
    ap.add_argument("-v", "--verbose", action="store_true", help="печатать ошибки")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    kwargs = {"model_name": args.model, "rerank": args.rerank}
    if args.ktru or "mock" in args.cases:
        kwargs["ktru_path"] = args.ktru or MOCK_PATH
    pipe = Pipeline(**kwargs)
    pipe.analyze("прогрев")  # первый запрос медленный из-за загрузки модели

    print(f"модель: {args.model}  реранкер: {args.rerank}  примеров: {len(cases)}\n")
    print("чистые тексты:   ", evaluate(pipe, cases, args.verbose))
    rng = random.Random(0)
    for i in range(args.noise_runs):
        noisy = [{**c, "text": add_noise(c["text"], rng)} for c in cases]
        print(f"с опечатками #{i + 1}:", evaluate(pipe, noisy, args.verbose))


if __name__ == "__main__":
    main()
