"""Evaluate error detection and typo correction on a hand-labeled regression set.

    python -m eval.errors
    python -m eval.errors --json
    python -m eval.errors --min-error-precision 0.8 --min-error-recall 0.8
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CASES = Path(__file__).with_name("error_cases.jsonl")
SCORED_ERROR_TYPES = {
    "conflict",
    "contradiction",
    "multiple_values",
    "nonexistent_value",
    "not_fitting_characteristic",
    "out_of_range",
    "product_not_found",
    "suspicious_value",
    "unit_mismatch",
    "uncertain_characteristic",
    "unreadable_fragment",
    "missing_value",
}
SCORED_CORRECTION_REASONS = ("опечат", "расклад", "уточнение названия")


def read_cases(path: Path = CASES) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(a: int, b: int) -> float:
    return round(a / b, 3) if b else 0.0


def _classification(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0}


def evaluate(pipe: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    error_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    total_tp = total_fp = total_fn = 0
    clean = clean_with_false_positive = 0
    correction_tp = correction_fp = correction_fn = correction_exact = correction_cases = 0
    failures = []

    for case in cases:
        result = pipe.analyze(case["text"])
        expected = set(case.get("expected_errors") or [])
        predicted = {error.get("type") for error in result.get("errors", [])
                     if error.get("type") in SCORED_ERROR_TYPES}
        if not expected:
            clean += 1
            clean_with_false_positive += bool(predicted)
        for kind in sorted(expected | predicted):
            if kind in expected and kind in predicted:
                error_counts[kind][0] += 1
                total_tp += 1
            elif kind in predicted:
                error_counts[kind][1] += 1
                total_fp += 1
            else:
                error_counts[kind][2] += 1
                total_fn += 1

        expected_corrections = {(item["from"].lower(), item["to"].lower())
                                for item in case.get("expected_corrections") or []}
        predicted_corrections = {
            (str(item.get("from", "")).lower(), str(item.get("to", "")).lower())
            for item in result.get("corrections", [])
            if any(reason in str(item.get("reason", "")).lower() for reason in SCORED_CORRECTION_REASONS)
        }
        if expected_corrections:
            correction_cases += 1
            correction_exact += expected_corrections == predicted_corrections
            correction_tp += len(expected_corrections & predicted_corrections)
            correction_fp += len(predicted_corrections - expected_corrections)
            correction_fn += len(expected_corrections - predicted_corrections)

        if expected != predicted or (expected_corrections and expected_corrections != predicted_corrections):
            failures.append({"id": case["id"], "expected_errors": sorted(expected),
                             "predicted_errors": sorted(predicted),
                             "expected_corrections": sorted(expected_corrections),
                             "predicted_corrections": sorted(predicted_corrections)})

    correction = _classification(correction_tp, correction_fp, correction_fn)
    correction["exact_case_accuracy"] = _ratio(correction_exact, correction_cases)
    correction["cases"] = correction_cases
    return {
        "n": len(cases),
        "errors_micro": _classification(total_tp, total_fp, total_fn),
        "errors_by_type": {kind: _classification(*counts) for kind, counts in sorted(error_counts.items())},
        "clean_false_positive_rate": _ratio(clean_with_false_positive, clean),
        "clean_cases": clean,
        "corrections": correction,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-error-precision", type=float, default=0.0)
    parser.add_argument("--min-error-recall", type=float, default=0.0)
    parser.add_argument("--min-correction-accuracy", type=float, default=0.0)
    args = parser.parse_args()

    from ml.pipeline import Pipeline
    from ml.service.analysis import AnalysisService
    from ml.service.holder import PipelineHolder

    holder = PipelineHolder(Pipeline)
    holder.load()
    if holder.status != "ready":
        raise RuntimeError(holder.error or "pipeline failed to load")
    service = AnalysisService(holder)
    service.warm_up()

    class Analyzer:
        @staticmethod
        def analyze(text: str) -> dict[str, Any]:
            return service.analyze({"raw": text, "answers": []})

    report = evaluate(Analyzer(), read_cases(args.cases))
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _text_report(report))
    failed = (report["errors_micro"]["precision"] < args.min_error_precision
              or report["errors_micro"]["recall"] < args.min_error_recall
              or report["corrections"]["exact_case_accuracy"] < args.min_correction_accuracy)
    return int(failed)


def _text_report(report: dict[str, Any]) -> str:
    lines = [f"examples: {report['n']}", f"errors micro: {report['errors_micro']}",
             f"clean false-positive rate: {report['clean_false_positive_rate']}",
             f"corrections: {report['corrections']}", "by type:"]
    lines += [f"  {kind}: {metrics}" for kind, metrics in report["errors_by_type"].items()]
    if report["failures"]:
        lines.append("mismatches:")
        lines += [f"  {item['id']}: expected={item['expected_errors']} predicted={item['predicted_errors']}"
                  for item in report["failures"]]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
