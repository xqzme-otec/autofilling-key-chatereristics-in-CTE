"""Семантический поиск позиций КТРУ по тексту пользователя.

1. Каждая позиция КТРУ (название + характеристики) заранее превращается в вектор.
2. Запрос превращается в вектор и сравнивается со всеми позициями (косинусная близость).
3. Опционально топ-K пересортировывает cross-encoder (реранкер) — точнее, но медленнее.

Векторы позиций кэшируются на диск, чтобы не пересчитывать при каждом старте.
"""
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from ml.ktru import KtruStore
from ml.text import tokenize

EMB_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

NAME_BOOST = 0.05  # прибавка к близости, если все значимые слова запроса есть в названии позиции

# модели семейства e5 обучены с префиксами "query: " / "passage: "
PREFIXES = {"e5": ("query: ", "passage: ")}


class Searcher:
    def __init__(self, store: KtruStore, model_name: str = EMB_MODEL, rerank: bool = False):
        self.store = store
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.q_prefix, self.p_prefix = next(
            (p for key, p in PREFIXES.items() if key in model_name.lower()), ("", ""))
        self.passages = [store.passage(it) for it in store.items]
        self.name_stems = [{t["lemma"][:5] for t in tokenize(it["name"]) if t["text"][0].isalpha()} for it in store.items]
        self.P = self._load_or_encode()
        self.reranker = CrossEncoder(RERANK_MODEL) if rerank else None

    def _load_or_encode(self) -> np.ndarray:
        key = hashlib.md5((self.model_name + "\n".join(self.passages)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        P = self.model.encode([self.p_prefix + p for p in self.passages],
                              normalize_embeddings=True, batch_size=64, show_progress_bar=len(self.passages) > 1000)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, P)
        return P

    def encode(self, texts: list[str], query: bool = True) -> np.ndarray:
        prefix = self.q_prefix if query else self.p_prefix
        return self.model.encode([prefix + t for t in texts], normalize_embeddings=True)

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
        sims = (self.P @ self.encode(queries).T).max(axis=1)
        sims = sims + NAME_BOOST * self._name_overlap(queries[-1])
        idx = np.argsort(-sims)[: (k_rerank if self.reranker else k)]
        scores = sims[idx]
        if self.reranker:
            scores = self.reranker.predict([(text, self.passages[i]) for i in idx])
            order = np.argsort(-scores)[:k]
            idx, scores = idx[order], scores[order]
        return [{"code": self.store.items[i]["code"], "name": self.store.items[i]["name"], "score": round(float(s), 4)}
                for i, s in zip(idx, scores)]
