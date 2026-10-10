"""Service layer of the model service: runs the pipeline, then the step the pipeline has no data for —
new characteristics seen in earlier descriptions (Postgres, ml/service/repository.py) — and logs what
goes into the model and what comes out.

A characteristic with origin "new" is in neither KTRU nor the portal dictionary. It gets
`known: true` when another description of the same product (category + product_intent) had it,
`known: false` otherwise. Seen for the first time, it is saved for the next requests only if it fits
the product (fits_product). With Postgres disabled or down the step is skipped: no `known` field, the
rest of the result is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import psycopg

from ml.service.holder import PipelineHolder
from ml.service.repository import NewCharacteristicsRepository
from ml.text import lemmatize_phrase

logger = logging.getLogger(__name__)

# after a Postgres failure the step is skipped this long instead of every request waiting for a connection
RETRY_AFTER_S = 30.0
LOG_TEXT_CHARS = 100  # AGENTS.md: at INFO a description is logged as its length and first 100 characters
LOG_VALUE_CHARS = 60
LOG_MAX_CHARACTERISTICS = 20


def characteristic_key(name: str) -> str:
    """Word forms and word order do not matter: «Рукоятка противоскользящая» = «противоскользящей рукоятки»."""
    return " ".join(sorted(lemmatize_phrase(name).split()))


def fits_product(category: str, product: str, name: str, value: str) -> bool:
    """Is a characteristic seen for the first time meaningful for this product? Only one that fits is
    saved and becomes known later; one that does not fit stays in the result (nothing is dropped,
    docs/DOC.md, invariant 2) but is not saved."""
    # TODO: decide with a model instead of accepting everything: e.g. e5 similarity (already loaded in
    #  the pipeline) of «name: value» to the KTRU descriptions of the product and to the portal
    #  characteristics of the category, threshold chosen on a benchmark (portal characteristics of the
    #  category vs ones from other categories). When it can say "no", also warn the user in `errors`.
    return True


class AnalysisService:
    def __init__(self, holder: PipelineHolder, repository: NewCharacteristicsRepository | None = None):
        self.holder = holder
        self.repository = repository
        self._skip_until = 0.0

    def analyze(self, request: dict[str, Any]) -> dict:
        _log_input(request)
        start = time.perf_counter()
        result, queued_ms = self.holder.analyze(request)
        pipeline_ms = (time.perf_counter() - start) * 1000 - queued_ms
        start = time.perf_counter()
        postgres = self._mark_known(result)
        postgres_ms = (time.perf_counter() - start) * 1000
        _log_output(request, result, postgres, queued_ms, pipeline_ms, postgres_ms)
        return result

    def _mark_known(self, result: dict) -> str:
        """Sets `known` on origin=new characteristics and saves the ones that fit the product; returns the
        step status for the log."""
        if self.repository is None:
            return "off"
        keyed = [(ch, characteristic_key(ch["name"])) for ch in result.get("characteristics") or []
                 if ch.get("origin") == "new"]
        keyed = [(ch, key) for ch, key in keyed if key]
        category, product = result.get("category"), result.get("product_intent")
        if not (keyed and category and product):
            return "-"
        if time.monotonic() < self._skip_until:
            return "skipped"
        text_hash = hashlib.sha256(result["text"].encode()).hexdigest()
        try:
            seen = self.repository.seen_before(category, product, [key for _, key in keyed], text_hash)
            # seen before = it fitted then; seen for the first time = it has to fit the product now
            accepted = {key: (key, ch["name"], ch["value"], ch.get("unit")) for ch, key in keyed
                        if key in seen or fits_product(category, product, ch["name"], ch["value"])}
            self.repository.save(category, product, text_hash, list(accepted.values()))
        except psycopg.Error as e:
            self._skip_until = time.monotonic() + RETRY_AFTER_S
            logger.warning("postgres unavailable, known new characteristics step skipped for %.0f s: %s",
                           RETRY_AFTER_S, e)
            return "failed"
        for ch, key in keyed:
            ch["known"] = key in seen
        logger.debug("new characteristics: category=%s product=%s seen_before=%s saved=%s not_fitting=%s",
                     category, product, seen, list(accepted), [key for _, key in keyed if key not in accepted])
        return "ok"


def _short(text: str, limit: int = LOG_VALUE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _describe(ch: dict) -> str:
    unit = f" {ch['unit']}" if ch.get("unit") else ""
    known = ", known" if ch.get("known") else ""
    return f"{_short(ch['name'])}={_short(str(ch['value']))}{unit} ({ch.get('origin')}{known})"


def _log_input(request: dict) -> None:
    answers = "; ".join(_short(f"{a['type']}:" + (f"{a['characteristic']}=" if a.get("characteristic") else "")
                               + a["value"]) for a in request["answers"])
    logger.info("analyze input: raw_len=%d raw=%r answers=%d [%s]", len(request["raw"]),
                _short(request["raw"], LOG_TEXT_CHARS), len(request["answers"]), answers)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("analyze input (full): %s", json.dumps(request, ensure_ascii=False))


def _log_output(request: dict, result: dict, postgres: str, queued_ms: float, pipeline_ms: float,
                postgres_ms: float) -> None:
    steps = result.get("steps") or []
    errors = result.get("errors") or []
    chars = result.get("characteristics") or []
    step = result.get("next_step") or {}
    logger.info(
        "analyze: raw_len=%d answers=%d category=%s product=%s search_candidates=%s candidates_total=%s top=%s "
        "ktru=%s characteristics=%d [%s] known=%d errors=%d error_types=%s next_step=%s postgres=%s "
        "queued_ms=%.1f pipeline_ms=%.1f postgres_ms=%.1f",
        len(request["raw"]), len(request["answers"]), result.get("category") or "-",
        result.get("product_intent") or "-", steps[0].get("count") if steps else 0, result.get("candidates_total"),
        ",".join(c["code"] for c in (result.get("candidates") or [])[:3]) or "-", result.get("ktru") or "none",
        len(chars), "; ".join(_describe(c) for c in chars[:LOG_MAX_CHARACTERISTICS]),
        sum(1 for c in chars if c.get("known")), len(errors),
        ",".join(sorted({str(e.get("type")) for e in errors})) or "-",
        f"{step.get('type')}:{step.get('characteristic') or '-'}({len(step.get('options') or [])} options)"
        if step else "none",
        postgres, queued_ms, pipeline_ms, postgres_ms)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("analyze output (full): %s", json.dumps(result, ensure_ascii=False, default=str))
