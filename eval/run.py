"""Quality evaluation of the ML part.

    python -m eval.run                         # clean and noisy texts
    python -m eval.run --model deepvk/USER-bge-m3 --rerank --cases eval/cases.jsonl --ktru data/ktru.json

Metrics:
  name@1 / name@5 — the correct product name is first / in the top 5 of the search;
  code_kept       — the correct code stayed among the candidates after narrowing (not lost);
  final_code_accuracy — the exact code after oracle answers to every clarifying question;
  wrong_final_code_rate — the dialog ended with another code;
  dialog_completion_rate — the dialog reached a code with the offered oracle answers;
  false_not_found — share of queries where the product was wrongly declared «not found»;
  chars P/R/F1    — precision (the value is allowed for the correct code) and recall (by the reference);
  latency         — mean and 95th percentile time per query.
"""
import argparse
import json
import random
import statistics
import time
from typing import Any

from eval.noise import add_noise
from ml.ktru import MOCK_PATH, format_value
from ml.pipeline import Pipeline
from ml.search import EMB_MODEL
from ml.text import lemmatize_phrase

MAX_DIALOG_TURNS = 12


def _answer_for_case(pipe: Pipeline, case: dict, question: dict) -> dict[str, str] | None:
    """Oracle answer from the labeled KTRU position, or None when the right option is unavailable."""
    options = question.get("options") or []
    if question.get("type") == "product":
        target = case.get("name") or pipe.store.by_code[case["ktru"]]["name"]
        return {"type": "product", "value": target} if target in options else None
    if question.get("type") == "code":
        return {"type": "code", "value": case["ktru"]} if case["ktru"] in options else None
    if question.get("type") != "characteristic" or not question.get("characteristic"):
        return None
    name = question["characteristic"]
    characteristic = next((ch for ch in pipe.store.by_code[case["ktru"]]["characteristics"]
                           if ch["name"] == name), None)
    if characteristic is None:
        return None
    target_options = [format_value(value, characteristic["unit"]) for value in characteristic["values"]]
    preferred = case.get("chars", {}).get(name)
    value = preferred if preferred in options else next((value for value in target_options if value in options), None)
    return {"type": "characteristic", "characteristic": name, "value": value} if value is not None else None


def resolve_dialog(pipe: Pipeline, case: dict, first: dict) -> dict[str, Any]:
    """Follow questions using the labeled position and report whether the dialog reaches a code."""
    result, answers, questions = first, [], 0
    answerable = True
    used_code_choice = False
    while not result.get("ktru") and questions < MAX_DIALOG_TURNS:
        question = result.get("next_step")
        if not question:
            answerable = False
            break
        answer = _answer_for_case(pipe, case, question)
        if answer is None:
            answerable = False
            break
        used_code_choice |= answer["type"] == "code"
        answers.append(answer)
        questions += 1
        result = pipe.analyze({"raw": case["text"], "answers": answers})
    return {
        "result": result,
        "questions": questions,
        "completed": answerable and bool(result.get("ktru")),
        "used_code_choice": used_code_choice,
    }


def evaluate(pipe: Pipeline, cases: list[dict], verbose: bool = False) -> dict:
    hit1 = hit5 = kept = not_found = tp = tp_p = n_pred = n_gold = 0
    exact_final = wrong_final = completed = code_choices = 0
    question_counts = []
    latencies = []
    for c in cases:
        want_name = c.get("name") or pipe.store.by_code[c["ktru"]]["name"]
        t = time.perf_counter()
        res = pipe.analyze(c["text"])
        latencies.append((time.perf_counter() - t) * 1000)

        # in KTRU one name has many codes — search is evaluated by name, the code is chosen by narrowing
        names = list(dict.fromkeys(x["name"] for x in pipe.searcher.search(res["text"], k=50)))
        hit1 += names[:1] == [want_name]
        hit5 += want_name in names[:5]
        kept += c["ktru"] in [x["code"] for x in res["candidates"]] + [res["ktru"]]
        not_found += any(e["type"] == "product_not_found" for e in res["errors"])

        dialog = resolve_dialog(pipe, c, res)
        final_result = dialog["result"]
        exact_final += final_result.get("ktru") == c["ktru"]
        wrong_final += bool(final_result.get("ktru")) and final_result.get("ktru") != c["ktru"]
        completed += dialog["completed"]
        code_choices += dialog["used_code_choice"]
        question_counts.append(dialog["questions"])

        # KTRU characteristics are evaluated; portal and new ones are outside KTRU, the code never has them
        pred = {(x["name"], x["value"]) for x in res["characteristics"]
                if x["method"] != "name" and x.get("origin", "ktru") == "ktru"}
        gold = set(c["chars"].items())
        # precision: the value is allowed for the correct code (the reference lacks some correct characteristics —
        # e.g. «Тип каркаса = Деревянный» from the name «Стул на деревянном каркасе»)
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
        "name@1": round(hit1 / n, 3), "name@5": round(hit5 / n, 3), "code_kept": round(kept / n, 3),
        "final_code_accuracy": round(exact_final / n, 3),
        "wrong_final_code_rate": round(wrong_final / n, 3),
        "dialog_completion_rate": round(completed / n, 3),
        "code_choice_rate": round(code_choices / n, 3),
        "questions_mean": round(statistics.mean(question_counts), 2),
        "questions_max": max(question_counts, default=0),
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
    pipe.analyze("прогрев")  # the first query is slow because the model loads

    print(f"модель: {args.model}  реранкер: {args.rerank}  примеров: {len(cases)}\n")
    print("чистые тексты:   ", evaluate(pipe, cases, args.verbose))
    rng = random.Random(0)
    for i in range(args.noise_runs):
        noisy = [{**c, "text": add_noise(c["text"], rng)} for c in cases]
        print(f"с опечатками #{i + 1}:", evaluate(pipe, noisy, args.verbose))


if __name__ == "__main__":
    main()
