from __future__ import annotations

"""Семантический поиск позиций КТРУ по тексту пользователя.

1. Каждая позиция КТРУ (название + описание + названия характеристик) заранее превращается в вектор.
2. Запрос превращается в вектор и сравнивается со всеми позициями (косинусная близость).
3. Опционально топ-K пересортировывает cross-encoder (реранкер) — точнее, но медленнее.

Векторы позиций кэшируются на диск, чтобы не пересчитывать при каждом старте.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from ml.ktru import KtruStore
from ml.text import tokenize

EMB_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "retrieval.json"
DOCUMENT_STRATEGY = "B_name_description_characteristic_names"
DOCUMENT_STRATEGY_VERSION = "2026-10-10"

NAME_BOOST = 0.05  # значение по умолчанию; runtime читает его из ml/config/retrieval.json

# модели семейства e5 обучены с префиксами "query: " / "passage: "
PREFIXES = {"e5": ("query: ", "passage: ")}


def load_retrieval_config(path: str | Path = CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {"name_boost": NAME_BOOST}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"name_boost": float(data.get("name_boost", NAME_BOOST))}


class Searcher:
    def __init__(self, store: KtruStore, model_name: str = EMB_MODEL, rerank: bool = False,
                 cache_dir: str | Path = CACHE_DIR, config_path: str | Path = CONFIG_PATH):
        self.store = store
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.config = load_retrieval_config(config_path)
        self.name_boost = self.config["name_boost"]
        self.model = SentenceTransformer(model_name)
        self.q_prefix, self.p_prefix = next(
            (p for key, p in PREFIXES.items() if key in model_name.lower()), ("", ""))
        self.passages = [store.passage(it) for it in store.items]
        self.name_stems = [{t["lemma"][:5] for t in tokenize(it["name"]) if t["text"][0].isalpha()} for it in store.items]
        self.P = self._load_or_encode()
        self.reranker = CrossEncoder(RERANK_MODEL) if rerank else None

    def _load_or_encode(self) -> np.ndarray:
        manifest = self._manifest()
        key = hashlib.md5(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        path = self.cache_dir / f"emb_{key}.npy"
        manifest_path = self.cache_dir / f"emb_{key}.manifest.json"
        if path.exists() and manifest_path.exists():
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self._manifest_matches(cached_manifest, manifest):
                return np.load(path)
        P = np.asarray(
            self.model.encode([self.p_prefix + p for p in self.passages],
                              normalize_embeddings=True, batch_size=64, show_progress_bar=len(self.passages) > 1000),
            dtype=np.float32,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, P)
        manifest["embedding_dim"] = int(P.shape[1]) if P.ndim == 2 else 0
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return P

    def _manifest(self) -> dict:
        return {
            "model_name": self.model_name,
            "document_strategy": DOCUMENT_STRATEGY,
            "document_strategy_version": DOCUMENT_STRATEGY_VERSION,
            "dataset_hash": self._dataset_hash(),
            # what is actually encoded: changes when loader filters change, even if the parquet file does not
            "passages_hash": hashlib.sha256("\n".join(self.passages).encode()).hexdigest(),
            "embedding_dim": None,
        }

    @staticmethod
    def _manifest_matches(cached: dict, expected: dict) -> bool:
        return all(cached.get(k) == v for k, v in expected.items() if k != "embedding_dim")

    def _dataset_hash(self) -> str:
        if getattr(self.store, "path", None) and Path(self.store.path).exists():
            h = hashlib.sha256()
            with Path(self.store.path).open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        payload = json.dumps(self.store.items, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def cache_path(self) -> Path:
        manifest = self._manifest()
        key = hashlib.md5(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir / f"emb_{key}.npy"

    def load_cached_embeddings(self) -> np.ndarray | None:
        path = self.cache_path()
        manifest_path = path.with_name(path.stem + ".manifest.json")
        if path.exists() and manifest_path.exists():
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self._manifest_matches(cached_manifest, self._manifest()):
                return np.load(path)
        return None

    def encode(self, texts: list[str], query: bool = True) -> np.ndarray:
        prefix = self.q_prefix if query else self.p_prefix
        return np.asarray(self.model.encode([prefix + t for t in texts], normalize_embeddings=True), dtype=np.float32)

    def _name_overlap(self, query: str) -> np.ndarray:
        """Доля значимых слов запроса, которые есть в названии позиции (по основе слова)."""
        q = {t["lemma"][:5] for t in tokenize(query) if t["text"][0].isalpha() and len(t["text"]) > 2}
        if not q:
            return np.zeros(len(self.name_stems))
        return np.array([len(q & n) / len(q) for n in self.name_stems])

    def search(self, text: str, k: int = 10, k_rerank: int = 50, extra_queries: list[str] = ()) -> list[dict]:
        """extra_queries — дополнительные формулировки (например, только название товара);
        близость позиции = максимум по всем формулировкам."""
        queries = list(dict.fromkeys([text, *[q for q in extra_queries if q.strip()]]))
        queries = [q for q in queries if q.strip()]
        if not queries or k <= 0:
            return []
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (self.P @ self.encode(queries).T).max(axis=1)
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        if self.name_boost:
            sims = sims + self.name_boost * self._name_overlap(queries[-1])
        idx = np.argsort(-sims)[: (k_rerank if self.reranker else k)]
        scores = sims[idx]
        if self.reranker:
            scores = self.reranker.predict([(text, self.passages[i]) for i in idx])
            order = np.argsort(-scores)[:k]
            idx, scores = idx[order], scores[order]
        return [{"code": self.store.items[i]["code"], "name": self.store.items[i]["name"], "score": round(float(s), 4)}
                for i, s in zip(idx, scores)]
