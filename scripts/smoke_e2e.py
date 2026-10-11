"""Five end-to-end demo scenarios against an already running Docker stack.

The script uses only the standard library and exits with a non-zero status on the
first broken contract. It does not start or modify the stack.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable


KTRU_CODE = re.compile(r"\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}")


class SmokeFailure(RuntimeError):
    pass


def request_json(url: str, payload: dict[str, Any] | None = None, request_id: str = "smoke") -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if data is None else "POST")
    request.add_header("Accept", "application/json")
    request.add_header("X-Request-ID", request_id)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"{request.method} {url}: HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise SmokeFailure(f"{request.method} {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{request.method} {url}: response is not JSON: {body[:500]}") from exc
    if not isinstance(parsed, dict):
        raise SmokeFailure(f"{request.method} {url}: expected a JSON object")
    return parsed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


class Stack:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def analyze(self, text: str, request_id: str) -> dict[str, Any]:
        result = request_json(f"{self.api_url}/api/v1/analyze", {"text": text}, request_id)
        require(isinstance(result.get("session_id"), str), "analyze response has no session_id")
        require(isinstance(result.get("characteristics"), list), "analyze response has no characteristics")
        require(isinstance(result.get("errors"), list), "analyze response has no errors")
        return result

    def answer(self, result: dict[str, Any], value: str, request_id: str) -> dict[str, Any]:
        session_id = result["session_id"]
        return request_json(
            f"{self.api_url}/api/v1/sessions/{session_id}/answers",
            {"value": value},
            request_id,
        )


def error_types(result: dict[str, Any]) -> set[str]:
    return {str(item.get("type")) for item in result["errors"]}


def clean_description_reaches_code(stack: Stack) -> dict[str, Any]:
    result = stack.analyze("стол письменный", "smoke-clean-1")
    questions = 0
    while not result.get("ktru") and questions < 10:
        question = result.get("next_step")
        require(isinstance(question, dict), "clean description stopped without a code or a question")
        options = question.get("options")
        require(isinstance(options, list) and options, "clarifying question has no options")
        questions += 1
        result = stack.answer(result, str(options[0]), f"smoke-clean-{questions + 1}")
    require(bool(KTRU_CODE.fullmatch(str(result.get("ktru") or ""))), "dialog did not reach a KTRU code")
    return {"code": result["ktru"], "questions": questions}


def typo_is_corrected(stack: Stack) -> dict[str, Any]:
    result = stack.analyze("стл на металлическом каркасе, без подлокотников", "smoke-typo")
    corrections = result.get("corrections") or []
    require(any(str(item.get("from", "")).lower() == "стл" for item in corrections),
            "the typo 'стл' was not reported as corrected")
    return {"corrections": len(corrections)}


def invalid_unit_is_explained(stack: Stack) -> dict[str, Any]:
    result = stack.analyze("молоток, длина рукоятки 2 кг", "smoke-unit")
    require("unit_mismatch" in error_types(result), "an incompatible unit was not reported")
    return {"errors": sorted(error_types(result))}


def ambiguity_requires_an_answer(stack: Stack) -> dict[str, Any]:
    result = stack.analyze("куртка мужская 48", "smoke-ambiguity-1")
    question = result.get("next_step")
    require(isinstance(question, dict), "ambiguous description did not produce a clarifying question")
    options = question.get("options")
    require(isinstance(options, list) and options, "ambiguous question has no options")
    before = result.get("remaining_count", len(result.get("candidates") or []))
    answered = stack.answer(result, str(options[0]), "smoke-ambiguity-2")
    after = answered.get("remaining_count", len(answered.get("candidates") or []))
    require(answered.get("ktru") is not None or answered.get("next_step") is not None,
            "answer stopped the dialog without a code or another question")
    require(not isinstance(before, int) or not isinstance(after, int) or after <= before,
            "answer increased the candidate pool")
    return {"question_type": question.get("type"), "before": before, "after": after}


def additional_characteristics_are_checked(stack: Stack) -> dict[str, Any]:
    accepted = stack.analyze("молоток слесарный, противоскользящая рукоятка", "smoke-new-valid")
    require(any("противоскольз" in str(item.get("name", "")).lower()
                and item.get("origin") in ("portal", "new") for item in accepted["characteristics"]),
            "a valid additional characteristic was not kept")
    rejected = stack.analyze("футболка мужская, мощность 800 Вт", "smoke-new-invalid")
    require("not_fitting_characteristic" in error_types(rejected),
            "a characteristic from another product domain was not explained")
    return {"valid_origins": sorted({str(item.get("origin")) for item in accepted["characteristics"]}),
            "invalid_errors": sorted(error_types(rejected))}


def wait_for_stack(api_url: str, model_url: str, wait_seconds: float,
                   allow_rules_only: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + wait_seconds
    last_error = "services have not answered yet"
    while True:
        try:
            backend = request_json(f"{api_url.rstrip('/')}/health", request_id="smoke-health-backend")
            model = request_json(f"{model_url.rstrip('/')}/ready", request_id="smoke-health-model")
            classifier = model.get("newchar_classifier")
            if backend.get("status") == "ok" and model.get("status") == "ready":
                if classifier == "loaded" or allow_rules_only and classifier == "rules_only":
                    return backend, model
                if classifier in ("rules_only", "load_failed"):
                    raise SmokeFailure(f"new-characteristic classifier is not loaded: {classifier}")
            last_error = f"backend={backend.get('status')} model={model.get('status')} classifier={classifier}"
        except SmokeFailure as exc:
            if "classifier is not loaded" in str(exc):
                raise
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise SmokeFailure(f"stack did not become ready in {wait_seconds:g}s: {last_error}")
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--model-url", default="http://localhost:8001")
    parser.add_argument("--wait-seconds", type=float, default=120)
    parser.add_argument("--allow-rules-only", action="store_true",
                        help="do not fail preflight when the new-characteristic classifier artifact is absent")
    args = parser.parse_args()

    backend, model = wait_for_stack(args.api_url, args.model_url, args.wait_seconds, args.allow_rules_only)

    stack = Stack(args.api_url)
    scenarios: list[tuple[str, Callable[[Stack], dict[str, Any]]]] = [
        ("clean_to_final_code", clean_description_reaches_code),
        ("typo_correction", typo_is_corrected),
        ("invalid_unit", invalid_unit_is_explained),
        ("ambiguity_dialog", ambiguity_requires_an_answer),
        ("additional_characteristics", additional_characteristics_are_checked),
    ]
    report = {"preflight": {"backend": backend, "model": model}, "scenarios": {}}
    for name, scenario in scenarios:
        report["scenarios"][name] = scenario(stack)
        print(f"PASS {name}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
