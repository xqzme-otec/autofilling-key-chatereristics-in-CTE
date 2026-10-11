"""Service layer of the model service: runs the pipeline, then the step the pipeline has no data for —
new characteristics seen in earlier descriptions (Postgres, ml/service/repository.py) — and logs what
goes into the model and what comes out.

A characteristic with origin "new" is in neither KTRU nor the portal dictionary. It gets
`known: true` when another description of the same product (category + product_intent) had it,
`known: false` otherwise. Seen for the first time, it is saved for the next requests only if it is valid for
the product: the rules of ml/fit.py, then the e5 + logistic regression classifier of ml/newchar.py. An invalid
one is not added to the card — only the warning with the reason stays; an uncertain one stays with a request to
confirm it; neither is saved. Without the classifier file only the rules decide. With Postgres disabled or down the remembering step
is skipped: no `known` field, the rest of the result is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import psycopg

from ml.fit import ProductFit
from ml.newchar import INVALID, MODEL_PATH, UNCERTAIN, VALID, NewCharClassifier
from ml.service.holder import PipelineHolder
from ml.service.repository import NewCharacteristicsRepository
from ml.text import lemmatize_phrase, main_noun

logger = logging.getLogger(__name__)

# after a Postgres failure the step is skipped this long instead of every request waiting for a connection
RETRY_AFTER_S = 30.0
LOG_TEXT_CHARS = 100  # AGENTS.md: at INFO a description is logged as its length and first 100 characters
LOG_VALUE_CHARS = 60
LOG_MAX_CHARACTERISTICS = 20


def characteristic_key(name: str) -> str:
    """Word forms and word order do not matter: «Рукоятка противоскользящая» = «противоскользящей рукоятки»."""
    return " ".join(sorted(lemmatize_phrase(name).split()))


class AnalysisService:
    def __init__(self, holder: PipelineHolder, repository: NewCharacteristicsRepository | None = None,
                 classifier_path: Path = MODEL_PATH):
        self.holder = holder
        self.repository = repository
        self.classifier_path = classifier_path
        self._skip_until = 0.0
        self._fit: ProductFit | None = None
        self._classifier: NewCharClassifier | None = None
        self._classifier_tried = False
        self._classifier_failed = False
        self._fit_lock = threading.Lock()

    def analyze(self, request: dict[str, Any]) -> dict:
        _log_input(request)
        start = time.perf_counter()
        result, queued_ms = self.holder.analyze(request)
        pipeline_ms = (time.perf_counter() - start) * 1000 - queued_ms
        start = time.perf_counter()
        not_fitting = self._check_fit(result)
        postgres = self._mark_known(result, not_fitting)
        postgres_ms = (time.perf_counter() - start) * 1000
        _log_output(request, result, postgres, queued_ms, pipeline_ms, postgres_ms)
        return result

    def warm_up(self) -> None:
        """Loads the rules and the classifier ahead of the first request (they need the loaded pipeline)."""
        if self._product_fit() is not None:
            self._new_char_classifier()

    def classifier_status(self) -> str:
        """Public readiness state without exposing the local artifact path."""
        if self.holder.status == "failed":
            return "load_failed"
        if self.holder.status != "ready":
            return "loading"
        if not self._classifier_tried:
            pipeline = self.holder.pipeline
            if getattr(pipeline, "store", None) is None \
                    or getattr(getattr(pipeline, "searcher", None), "model", None) is None:
                return "rules_only"
            return "loading" if self.classifier_path.exists() else "rules_only"
        if self._classifier is not None:
            return "loaded"
        return "load_failed" if self._classifier_failed else "rules_only"

    def _product_fit(self) -> ProductFit | None:
        """Built once from the loaded pipeline's KTRU store; None for a pipeline without one (tests)."""
        if self._fit is None:
            store = getattr(self.holder.pipeline, "store", None)
            if store is None:
                return None
            with self._fit_lock:
                if self._fit is None:
                    self._fit = ProductFit(store)
        return self._fit

    def _new_char_classifier(self) -> NewCharClassifier | None:
        """Loaded once with the pipeline's e5 model; None without the model file (only the rules decide)."""
        if not self._classifier_tried:
            with self._fit_lock:
                if not self._classifier_tried:
                    pipeline = self.holder.pipeline
                    encoder = getattr(getattr(pipeline, "searcher", None), "model", None)
                    store = getattr(pipeline, "store", None)
                    if encoder is not None and store is not None:
                        if not self.classifier_path.exists():
                            logger.warning("no newchar classifier at %s (python -m ml.newchar build): "
                                           "new characteristics are judged by the rules only", self.classifier_path)
                        else:
                            try:
                                self._classifier = NewCharClassifier(encoder, store, self.classifier_path,
                                                                     fit=self._fit)
                            except Exception:
                                self._classifier_failed = True
                                logger.exception("newchar classifier failed to load: rules only")
                    self._classifier_tried = True
        return self._classifier

    def _check_fit(self, result: dict) -> set[str]:
        """Rules first (ml/fit.py), then the classifier (ml/newchar.py) for origin=new characteristics. Invalid
        ones are removed from the card and get a warning, uncertain ones stay with a request to confirm. Returns
        the names of those that must not be remembered. An origin=portal characteristic gets a warning when none of the
        product's portal cards has it: the portal dictionary is shared by a whole category."""
        category, product = result.get("category"), result.get("product_intent")
        chars = result.get("characteristics") or []
        new = [ch for ch in chars if ch.get("origin") == "new"]
        portal = [ch for ch in chars if ch.get("origin") == "portal"]
        fit = self._product_fit() if (new or portal) and category and product else None
        if fit is None:
            return set()
        errors = result.setdefault("errors", [])
        classifier = self._new_char_classifier()
        shown = result.get("ktru_name") or product
        for ch in portal:
            if main_noun(str(ch["value"])) == product:
                continue  # «куртка» -> «Вид санитарной одежды: Куртка»: the product itself, not a user's characteristic
            # a dimension the product never has («молоток, объем двигателя 2 л» — portal: chainsaws) or a word of
            # another category («витрина, тип изделия Пассатижи»)
            reason = fit.reason(category, product, ch["name"], str(ch["value"]))
            if reason:
                errors.append(_not_fitting(ch["name"], f"{reason}.", "rules"))
                continue
            doubt = fit.doubt(category, product, ch["name"], str(ch["value"]))
            if doubt:
                errors.append(_uncertain(ch["name"], f"{doubt}.", "rules"))
                continue
            cards = classifier.portal_cards_without(category, product, ch["name"]) if classifier is not None else None
            # absent from the product's cards: a measured value is foreign («дрель, разрешение экрана 4K», «жакет,
            # объем 1 литр»); a word value only when the classifier rejects it too — «платье, с капюшоном» and
            # «халат, вышивка на кармане» are absent from the cards but fit
            if not cards:
                continue
            numeric = any(c.isdigit() for c in str(ch["value"]))
            zone, p_valid = classifier.predict([{"category": category, "product": product, "name": ch["name"],
                                                 "value": str(ch["value"])}])[0]
            absent = f"«{ch['name']}» не встречается ни в одной из {cards} карточек товаров «{product}» на Портале поставщиков."
            if zone == INVALID or (numeric and zone == UNCERTAIN):
                errors.append(_not_fitting(ch["name"], absent, "portal_cards", p_valid))
            elif numeric:
                # fits by meaning, but the cards call it otherwise or never fill it in («шуруповерт, частота
                # вращения» — «число оборотов» in the cards): ask, do not reject
                errors.append({"type": "uncertain_characteristic", "severity": "info", "characteristic": ch["name"],
                               "checked_by": "portal_cards", "confidence": p_valid,
                               "message": f"{absent} Подтвердите, что она относится к товару."})
        not_saved: dict[str, str] = {}  # name -> zone (checked by)
        rest = []
        for ch in new:
            reason = fit.reason(category, product, ch["name"], str(ch["value"]))
            doubt = None if reason else fit.doubt(category, product, ch["name"], str(ch["value"]))
            if reason:
                not_saved[ch["name"]] = "invalid (rules)"
                errors.append(_not_fitting(ch["name"], f"{reason}.", "rules"))
            elif doubt:
                not_saved[ch["name"]] = "uncertain (rules)"
                errors.append(_uncertain(ch["name"], f"{doubt}.", "rules"))
            else:
                rest.append(ch)
        if classifier is not None and rest:
            zones = classifier.predict([{"category": category, "product": product, "name": ch["name"],
                                         "value": str(ch["value"])} for ch in rest])
            for ch, (zone, p_valid) in zip(rest, zones):
                if zone == INVALID:
                    not_saved[ch["name"]] = f"invalid ({p_valid})"
                    errors.append(_not_fitting(ch["name"], f"«{ch['name']}» не похоже на характеристику товара "
                                                           f"«{shown}».", "model", p_valid))
                elif zone == UNCERTAIN:
                    not_saved[ch["name"]] = f"uncertain ({p_valid})"
                    errors.append({"type": "uncertain_characteristic", "severity": "info", "characteristic": ch["name"],
                                   "checked_by": "model", "confidence": p_valid,
                                   "message": f"Не уверены, что «{ch['name']}» относится к товару «{shown}». "
                                              "Подтвердите или исправьте; пока признак не запоминается для категории."})
        flagged = [e["characteristic"] for e in errors if e.get("checked_by") == "portal_cards"]
        if flagged:
            logger.warning("portal characteristics absent from the product's portal cards: category=%s product=%s %s",
                           category, product, flagged)
        if not_saved:
            logger.warning("new characteristics not remembered: category=%s product=%s %s", category, product,
                           sorted(f"{name}: {zone}" for name, zone in not_saved.items()))
        # an invalid characteristic is not added to the card: only the warning with the reason stays
        rejected = {e["characteristic"] for e in errors if e["type"] == "not_fitting_characteristic"}
        result["characteristics"] = [ch for ch in chars if not (ch.get("origin") in ("portal", "new")
                                                                and ch["name"] in rejected)]
        # «сохранены дополнительные признаки» must not list what got a warning or a request to confirm
        _drop_from_not_in_ktru(errors, rejected | {e["characteristic"] for e in errors
                                                   if e["type"] == "uncertain_characteristic"})
        return set(not_saved)

    def _mark_known(self, result: dict, not_fitting: set[str] = frozenset()) -> str:
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
                        if key in seen or ch["name"] not in not_fitting}
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


NOT_IN_KTRU_MESSAGE = re.compile(r"^(Для «.*?» сохранены дополнительные признаки: )(.*)(\. Они не исключают коды КТРУ\.)$")


def _drop_from_not_in_ktru(errors: list[dict], rejected: set[str]) -> None:
    """The not_in_ktru notice says the listed features were kept; the ones that got a not_fitting warning do not
    belong there. Without any left the notice is removed."""
    if not rejected:
        return
    for error in [e for e in errors if e["type"] == "not_in_ktru"]:
        error["characteristics"] = [n for n in error.get("characteristics") or [] if n not in rejected]
        m = NOT_IN_KTRU_MESSAGE.match(error.get("message", ""))
        listed = [n for n in re.findall(r"«([^»]+)»", m.group(2)) if n not in rejected] if m else None
        if not error["characteristics"] or listed == []:
            errors.remove(error)
        elif m:
            error["message"] = m.group(1) + ", ".join(f"«{n}»" for n in listed) + m.group(3)


def _uncertain(name: str, reason: str, checked_by: str, confidence: float | None = None) -> dict:
    error = {"type": "uncertain_characteristic", "severity": "info", "characteristic": name, "checked_by": checked_by,
             "message": f"{reason} Подтвердите, что признак относится к товару; пока он не запоминается для категории."}
    if confidence is not None:
        error["confidence"] = confidence
    return error


def _not_fitting(name: str, reason: str, checked_by: str, confidence: float | None = None) -> dict:
    error = {"type": "not_fitting_characteristic", "severity": "warning", "characteristic": name,
             "checked_by": checked_by,
             "message": f"{reason} Признак не добавлен в карточку товара; проверьте описание."}
    if confidence is not None:
        error["confidence"] = confidence  # P(valid) from the classifier
    return error


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
