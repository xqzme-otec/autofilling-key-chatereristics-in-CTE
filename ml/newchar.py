"""Is a new characteristic (in neither KTRU nor the portal dictionary) valid for the product?

After the rules of ml/fit.py, a logistic regression over multilingual-e5-small features (the model the pipeline
has already loaded) gives P(valid) for (product, characteristic, value), and two thresholds split it into
three zones: VALID (remember it), INVALID (warn, do not remember), UNCERTAIN (ask the user to confirm).

Training pairs come from real supplier portal cards (sheet 1 of the exports):
  valid   — a characteristic of a card, with the card's own product («стол» — «Высота опор 725 мм»);
  invalid — the same characteristic with a product of another category whose portal dictionary does not have
            it («кресло» — «Длина волны»), plus evaluative words with any product («Стильный»).
Characteristic names of the validation part never occur in training. The thresholds are chosen on the
validation part plus the hand-labeled eval/newchar_val.jsonl: each decided zone must be right in 98 % of cases.
On the hand-written eval/newchar_manual.jsonl, rules have macro-F1 0.873 and e5 + LR has 0.838 because it sends
more rare valid cases to UNCERTAIN; it also sends 7 of 10 ambiguous cases there (see ML.md). The classifier is a
semantic safety layer, not a blanket replacement for rules. Heavy generative LLMs were rejected: the production server
has 4 CPU and 4 GB RAM.

    .venv/bin/python -m ml.newchar build   # data/raw/supplier_portal/*.xlsx -> data/processed/newchar_model.pkl
    .venv/bin/python -m ml.newchar eval    # metrics on eval/newchar_manual.jsonl
"""
from __future__ import annotations

import json
import logging
import pickle
import random
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.fit import SUBJECTIVE, SUBJECTIVE_ONLY, ProductFit
from ml.portal import DICT_PATH
from ml.text import lemmatize_phrase, main_noun

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "supplier_portal"
MODEL_PATH = ROOT / "data" / "processed" / "newchar_model.pkl"
VAL_PATH = ROOT / "eval" / "newchar_val.jsonl"
MANUAL_PATH = ROOT / "eval" / "newchar_manual.jsonl"
FILES = {"Инструменты": "инструменты", "Мебель": "мебель", "Одежда": "одежда"}
VERSION = 1

CARDS_PER_FILE = 30000      # sampled cards per export file
MIN_PRODUCT_CARDS = 50      # a portal characteristic is suspicious for a product only with this many cards to judge by
PAIRS_PER_CATEGORY = 1500   # valid pairs per category (each gets one foreign invalid pair)
SUBJECTIVE_PAIRS = 12       # pairs per evaluative word
VAL_SHARE = 0.2             # share of characteristic names held out for validation
TARGET_PRECISION = 0.98     # each decided zone must be right this often on validation
C_GRID = (0.03, 0.1, 0.3, 1.0)
MIN_PORTAL_USAGE = 3        # portal characteristics used in fewer KPGZ categories are not reference names
SKIP = re.compile(r"^(бренд|артикул|модель|производитель|страна|торговая|гарант|срок|код|ean|штрих|описани|серия|"
                  r"тип товара|вид товара|наименован|вид товаров|виды одежды)", re.IGNORECASE)
VALID, INVALID, UNCERTAIN = "valid", "invalid", "uncertain"


def char_text(name: str, value: str) -> str:
    value = str(value or "")
    return name if value in ("", "Да") else f"{name} {value}"


class Features:
    """e5 embeddings of (product, characteristic) + similarities to reference characteristic names + rule flags.
    The same code builds training and runtime features."""

    def __init__(self, encoder, store, portal: dict, fit: ProductFit):
        self.encoder = encoder
        self.fit = fit
        ref: dict[tuple, set] = defaultdict(set)
        for item in store.items:
            for ch in item["characteristics"]:
                ref[("p", item["category"], main_noun(item["name"]))].add(ch["name"])
                ref[("c", item["category"])].add(ch["name"])
        for category, chars in portal.items():
            ref[("c", category)] |= {n for n, info in chars.items()
                                     if not n.startswith("_") and info.get("usage", 0) >= MIN_PORTAL_USAGE}
        self.categories = sorted(k[1] for k in ref if k[0] == "c")
        names = sorted({n for names in ref.values() for n in names})
        self.names = self.encode(names, "passage: ")
        index = {n: i for i, n in enumerate(names)}
        self.ref = {k: np.array([index[n] for n in v]) for k, v in ref.items()}

    def encode(self, texts: list[str], prefix: str = "query: ") -> np.ndarray:
        return np.asarray(self.encoder.encode([prefix + t for t in texts], normalize_embeddings=True,
                                              batch_size=128, show_progress_bar=False), dtype=np.float32)

    def __call__(self, items: list[dict]) -> np.ndarray:
        """items: [{"category", "product", "name", "value"}]"""
        texts = [char_text(it["name"], it["value"]) for it in items]
        pair = self.encode([f"товар: {it['product']}. характеристика: {t}" for it, t in zip(items, texts)])
        char = self.encode(texts)
        product = self.encode([it["product"] for it in items])
        sims = char @ self.names.T
        scalars = []
        for i, it in enumerate(items):
            category = it["category"] if it["category"] in self.categories else self.categories[0]
            in_category = sims[i, self.ref[("c", category)]].max()
            own_key = ("p", category, it["product"])
            own = sims[i, self.ref[own_key]].max() if own_key in self.ref else in_category
            other = max(sims[i, self.ref[("c", c)]].max() for c in self.categories if c != category)
            reason = self.fit.reason(category, it["product"], it["name"], str(it["value"] or ""))
            scalars.append([float(char[i] @ product[i]), own, in_category, other, own - other, in_category - other,
                            float(bool(reason) and "субъектив" in reason), float(bool(reason) and "не бывает" in reason)]
                           + [float(category == c) for c in self.categories])
        return np.hstack([pair, char * product, np.array(scalars, dtype=np.float32)])


class NewCharClassifier:
    """Runtime: P(valid) and the zone for new characteristics; built from the pipeline's encoder and KTRU store."""

    def __init__(self, encoder, store, model_path: Path | None = None, portal_path: Path | None = None,
                 fit: ProductFit | None = None):
        start = time.perf_counter()
        saved = pickle.loads(Path(model_path or MODEL_PATH).read_bytes())
        if saved.get("version") != VERSION:
            raise ValueError(f"newchar model version {saved.get('version')}, expected {VERSION}: rebuild it")
        portal_path = portal_path or DICT_PATH
        portal = json.loads(portal_path.read_text(encoding="utf-8")) if portal_path.exists() else {}
        self.model, self.lo, self.hi = saved["model"], saved["lo"], saved["hi"]
        self.product_cards = saved.get("product_cards", {})
        self.features = Features(encoder, store, portal, fit or ProductFit(store, portal_path))
        self._lock = threading.Lock()
        logger.info("newchar classifier loaded in %.2f s: thresholds invalid<=%.3f valid>=%.3f, built %s",
                    time.perf_counter() - start, self.lo, self.hi, saved.get("built"))

    def portal_cards_without(self, category: str, product: str, name: str) -> int | None:
        """Number of portal cards of the product when none of them has this characteristic, else None.
        «стол» — «Разрешение экрана»: the name is in the furniture portal dictionary (interactive panels), but
        not in any of the table cards."""
        cards = self.product_cards.get(category, {}).get(product)
        if cards and cards[0] >= MIN_PRODUCT_CARDS and not cards[1].get(name):
            return cards[0]
        return None

    def zone(self, p_valid: float) -> str:
        return VALID if p_valid >= self.hi else INVALID if p_valid <= self.lo else UNCERTAIN

    def predict(self, items: list[dict]) -> list[tuple[str, float]]:
        """items: [{"category", "product", "name", "value"}] -> [(zone, P(valid))]"""
        if not items:
            return []
        with self._lock:
            p = self.model.predict_proba(self.features(items))[:, 1]
        return [(self.zone(float(x)), round(float(x), 3)) for x in p]


# --- offline: training data, training, evaluation

def _card_pairs(store, portal: dict) -> tuple[list[dict], dict]:
    """Training pairs and, per category and product, the number of cards and of cards with each characteristic."""
    import pandas as pd

    rng = random.Random(11)
    products: dict[str, set] = defaultdict(set)
    for item in store.items:
        noun = main_noun(item["name"])
        if noun:
            products[item["category"]].add(noun)
    names_by_cat = {c: sorted(((lemmatize_phrase(n).split(), n) for n in chars if not n.startswith("_")),
                              key=lambda x: -len(x[0])) for c, chars in portal.items()}
    rows: dict[str, list] = {}
    product_cards: dict[str, dict] = defaultdict(dict)
    for file, category in FILES.items():
        df = pd.read_excel(RAW_DIR / f"{file}.xlsx", sheet_name=0, dtype=str,
                           usecols=["Наименование СТЕ", "Характеристики"])
        df = df.sample(min(len(df), CARDS_PER_FILE), random_state=2)
        found = []
        for name, chars in df.itertuples(index=False):
            product = main_noun(str(name))
            if product not in products[category] or not isinstance(chars, str):
                continue
            card = product_cards[category].setdefault(product, [0, defaultdict(int)])
            card[0] += 1
            card_names = set()
            for raw in (x.strip() for x in chars.split(" | ")):
                if not raw or SKIP.match(raw) or len(raw) > 70:
                    continue
                lemmas = lemmatize_phrase(raw).split()
                hit = next((n for nl, n in names_by_cat[category] if lemmas[:len(nl)] == nl and len(lemmas) > len(nl)), None)
                if hit:
                    found.append((product, hit, " ".join(raw.split()[len(hit.split()):]).strip() or "Да"))
                    card_names.add(hit)
            for hit in card_names:
                card[1][hit] += 1
        rows[category] = found
        logger.info("%s: %d card characteristics", file, len(found))
    pairs, seen = [], set()
    for category, found in rows.items():
        rng.shuffle(found)
        others = [c for c in rows if c != category]
        n = 0
        for product, name, value in found:
            if (product, name) in seen:
                continue
            seen.add((product, name))
            pairs.append({"category": category, "product": product, "name": name, "value": value, "label": 1,
                          "group": name.lower()})
            foreign = [c for c in others if name not in portal.get(c, {})]
            if foreign:
                other = rng.choice(foreign)
                pairs.append({"category": other, "product": rng.choice(sorted(products[other])), "name": name,
                              "value": value, "label": 0, "group": name.lower()})
            n += 1
            if n >= PAIRS_PER_CATEGORY:
                break
    for word in sorted(SUBJECTIVE - SUBJECTIVE_ONLY):
        for _ in range(SUBJECTIVE_PAIRS):
            category = rng.choice(sorted(products))
            pairs.append({"category": category, "product": rng.choice(sorted(products[category])),
                          "name": word.capitalize(), "value": "Да", "label": 0, "group": "subjective:" + word})
    cards = {c: {p: (n, dict(names)) for p, (n, names) in by_product.items() if n >= MIN_PRODUCT_CARDS}
             for c, by_product in product_cards.items()}
    return pairs, cards


def thresholds(scores: np.ndarray, y: np.ndarray, target: float = TARGET_PRECISION) -> tuple[float, float]:
    """The widest VALID (score >= hi) and INVALID (score <= lo) zones that are each right >= target."""
    order = np.unique(scores)
    hi = next((t for t in order if (y[scores >= t] == 1).mean() >= target), np.inf)
    lo = next((t for t in order[::-1] if (y[scores <= t] == 0).mean() >= target), -np.inf)
    if lo >= hi:
        lo = hi = (lo + hi) / 2
    return float(lo), float(hi)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _encoder_and_store():
    from sentence_transformers import SentenceTransformer

    from ml.ktru import KtruStore
    from ml.search import EMB_MODEL
    return SentenceTransformer(EMB_MODEL), KtruStore()


def build() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    encoder, store = _encoder_and_store()
    portal = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    pairs, cards = _card_pairs(store, portal)
    groups = sorted({p["group"] for p in pairs})
    val_groups = set(random.Random(5).sample(groups, int(len(groups) * VAL_SHARE)))
    train = [p for p in pairs if p["group"] not in val_groups]
    val = [p for p in pairs if p["group"] in val_groups] + (_read_jsonl(VAL_PATH) if VAL_PATH.exists() else [])
    features = Features(encoder, store, portal, ProductFit(store))
    x_train, y_train = features(train), np.array([p["label"] for p in train])
    x_val, y_val = features(val), np.array([p["label"] for p in val])
    best = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(x_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        print(f"C={c}: validation AUC {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, c, model)
    auc, c, model = best
    lo, hi = thresholds(model.predict_proba(x_val)[:, 1], y_val)
    MODEL_PATH.write_bytes(pickle.dumps({"version": VERSION, "model": model, "lo": lo, "hi": hi, "C": c,
                                         "product_cards": cards,
                                         "val_auc": auc, "built": time.strftime("%Y-%m-%d %H:%M")}))
    print(f"train {len(train)} pairs, validation {len(val)}; C={c} AUC {auc:.3f}; "
          f"thresholds invalid<={lo:.3f} valid>={hi:.3f} -> {MODEL_PATH}")


def evaluate(path: Path = MANUAL_PATH) -> dict:
    """Rules alone vs rules + classifier on hand-labeled cases (label valid / invalid / ambiguous)."""
    encoder, store = _encoder_and_store()
    clf = NewCharClassifier(encoder, store)
    cases = _read_jsonl(path)
    for case in cases:
        case.setdefault("category", next((it["category"] for it in store.items
                                          if main_noun(it["name"]) == case["product"]), "одежда"))
    rules = [bool(clf.features.fit.reason(c["category"], c["product"], c["name"], c["value"])) for c in cases]
    zones = [INVALID if r else z for r, (z, _) in zip(rules, clf.predict(cases))]
    out = {}
    for method, predicted in (("rules", [INVALID if r else VALID for r in rules]), ("rules+e5+LR", zones)):
        labeled = [(c["label"], z) for c, z in zip(cases, predicted) if c["label"] != "ambiguous"]
        f1s = []
        for cls in (VALID, INVALID):
            tp = sum(1 for y, z in labeled if y == cls and z == cls)
            precision, recall = tp / max(sum(z == cls for _, z in labeled), 1), tp / max(sum(y == cls for y, _ in labeled), 1)
            f1s.append(2 * precision * recall / max(precision + recall, 1e-9))
            out[f"{method}.{cls}_precision"] = round(precision, 3)
        out[f"{method}.macro_f1"] = round(sum(f1s) / 2, 3)
        out[f"{method}.uncertain"] = round(sum(z == UNCERTAIN for _, z in labeled) / len(labeled), 3)
        out[f"{method}.ambiguous_to_uncertain"] = sum(1 for c, z in zip(cases, predicted)
                                                      if c["label"] == "ambiguous" and z == UNCERTAIN)
    for key, value in out.items():
        print(f"{key}: {value}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if sys.argv[1:] == ["build"]:
        build()
    elif sys.argv[1:] == ["eval"]:
        evaluate()
    else:
        print(__doc__)
