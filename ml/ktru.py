from __future__ import annotations

"""Загрузка справочника КТРУ: data/processed/ktru.parquet (парсер ml/data/) или JSON временных данных."""
import json
from collections import Counter
from pathlib import Path

from ml.text import lemma, tokenize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PATH = DATA_DIR / "processed" / "ktru.parquet"
MOCK_PATH = DATA_DIR / "mock" / "ktru_mock.json"
CATEGORY_RU = {"clothing": "одежда", "furniture": "мебель", "tools": "инструменты"}


def _num(x) -> float | int | None:
    if x is None:
        return None
    f = float(str(x).replace(",", "."))
    return int(f) if f.is_integer() else f


def _convert_char(c: dict) -> dict | None:
    """Характеристика из формата парсера ml/data → {name, required, unit, values}."""
    values, unit = [], None
    for v in c["values"]:
        unit = unit or v.get("okei_name")
        if v.get("quality_description"):
            values.append({"text": v["quality_description"].strip()})
        for x in v.get("concrete_values") or []:
            try:
                n = _num(x)
                values.append({"min": n, "max": n})
            except ValueError:
                values.append({"text": str(x).strip()})
        for g in v.get("ranges") or []:
            r = {}
            if g.get("min_value") is not None:
                r["min"] = _num(g["min_value"])
                r["min_incl"] = g.get("min_notation") != "greater"
            if g.get("max_value") is not None:
                r["max"] = _num(g["max_value"])
                r["max_incl"] = g.get("max_notation") != "less"
            if r:
                values.append(r)
    if not c.get("name") or not values:
        return None
    return {"name": c["name"].strip(), "required": bool(c.get("is_required")), "unit": unit, "values": values}


def load_parquet(path: Path, active_only: bool = True, skip_templates: bool = True) -> list[dict]:
    """Читает data/processed/ktru.parquet (результат scripts/prepare_ktru_dataset.py).

    Шаблоны (is_template) — родительские позиции, а не конечные коды; TERMINATED — недействующие.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if active_only:
        df = df[df["status"] == "ACTIVE"]
    if skip_templates:
        df = df[~df["is_template"].astype(bool)]
    items = []
    for r in df.itertuples(index=False):
        chars = [ch for ch in map(_convert_char, json.loads(r.characteristics_json or "[]")) if ch]
        okeis = json.loads(r.position_okeis_json or "[]")
        parent = json.loads(r.parent_position_info_json or "null")
        items.append({
            "code": r.code,
            "name": (r.name or "").strip(),
            # описание есть у единиц позиций, у остальных в parquet NaN (float) — берём только строки
            "description": r.nsi_description.strip() if isinstance(getattr(r, "nsi_description", None), str) else "",
            "okpd2": r.okpd2_code,
            "category": CATEGORY_RU.get(r.category, r.category),
            "unit": okeis[0]["name"] if okeis else None,
            "parent": parent.get("code") if parent else None,
            "characteristics": chars,
        })
    return items


def is_text_char(ch: dict) -> bool:
    return any("text" in v for v in ch["values"])


def is_bool_char(ch: dict) -> bool:
    texts = {v.get("text") for v in ch["values"]}
    return bool(texts) and texts <= {"Да", "Нет"}


def format_value(v: dict, unit: str | None = None) -> str:
    if "text" in v:
        return v["text"]
    lo, hi = v.get("min"), v.get("max")
    if lo is not None and hi is not None:
        s = f"{lo}" if lo == hi else f"{lo}–{hi}"
    elif hi is not None:
        s = f"{'≤' if v.get('max_incl') else '<'} {hi}"
    else:
        s = f"{'≥' if v.get('min_incl', True) else '>'} {lo}"
    return f"{s} {unit}" if unit else s


class KtruStore:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        if self.path.suffix == ".parquet":
            self.items = load_parquet(self.path)
        else:
            self.items = json.loads(self.path.read_text(encoding="utf-8"))
        self.by_code = {it["code"]: it for it in self.items}
        self.vocab = self._build_vocab()

    def passage(self, item: dict) -> str:
        """Текст позиции для эмбеддингов: название + описание + названия характеристик."""
        parts = [f"Наименование: {item['name']}."]
        if item.get("description"):
            parts.append(f"Описание: {item['description']}.")
        names = [ch["name"] for ch in item["characteristics"]]
        if names:
            parts.append("Характеристики: " + "; ".join(dict.fromkeys(names)) + ".")
        return " ".join(parts)

    def _build_vocab(self) -> Counter:
        """Частоты слов КТРУ (словоформы и леммы) — словарь для исправления опечаток."""
        vocab = Counter()
        for it in self.items:
            texts = [it["name"]]
            for ch in it["characteristics"]:
                texts.append(ch["name"])
                texts += [v["text"] for v in ch["values"] if "text" in v]
            for text in texts:
                for t in tokenize(text):
                    if t["text"][0].isalpha() and len(t["text"]) > 1:
                        vocab[t["text"].lower()] += 1
                        vocab[lemma(t["text"])] += 1
        return vocab
