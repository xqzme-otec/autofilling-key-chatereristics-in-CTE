import json
from pathlib import Path

import numpy as np

import ml.search as search
from ml.ktru import MOCK_PATH
from ml.pipeline import Pipeline


class DummyModel:
    document_encode_calls = 0

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True, batch_size=None, show_progress_bar=False):
        if len(texts) > 1:
            DummyModel.document_encode_calls += 1
        vectors = np.asarray([self._vector(t) for t in texts], dtype=np.float32)
        if normalize_embeddings:
            norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norm, 1e-9)
        return vectors

    @staticmethod
    def _vector(text):
        text = text.lower()
        if "стул" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "молот" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "шкаф" in text:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


class TinyStore:
    def __init__(self, path: Path):
        self.path = path
        self.items = [
            {"code": "chair", "name": "Стул офисный", "characteristics": [{"name": "Материал обивки"}]},
            {"code": "hammer", "name": "Молоток", "characteristics": [{"name": "Масса молотка"}]},
            {"code": "wardrobe", "name": "Шкаф", "characteristics": [{"name": "Материал корпуса"}]},
        ]

    def passage(self, item):
        names = "; ".join(ch["name"] for ch in item["characteristics"])
        return f"Наименование: {item['name']}. Характеристики: {names}."


def _config(path: Path, name_boost=0.0):
    path.write_text(json.dumps({"name_boost": name_boost}), encoding="utf-8")
    return path


def _store(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("dataset-v1", encoding="utf-8")
    return TinyStore(dataset_path)


def test_search_respects_top_k(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SentenceTransformer", DummyModel)
    s = search.Searcher(_store(tmp_path), cache_dir=tmp_path, config_path=_config(tmp_path / "retrieval.json"))

    result = s.search("стул", k=2)

    assert len(result) == 2
    assert result[0]["code"] == "chair"


def test_search_ranking_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SentenceTransformer", DummyModel)
    s = search.Searcher(_store(tmp_path), cache_dir=tmp_path, config_path=_config(tmp_path / "retrieval.json"))

    first = s.search("молоток", k=3)
    second = s.search("молоток", k=3)

    assert first == second


def test_empty_query_returns_no_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SentenceTransformer", DummyModel)
    s = search.Searcher(_store(tmp_path), cache_dir=tmp_path, config_path=_config(tmp_path / "retrieval.json"))

    assert s.search("", k=3) == []


def test_unknown_query_still_returns_top_k(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SentenceTransformer", DummyModel)
    s = search.Searcher(_store(tmp_path), cache_dir=tmp_path, config_path=_config(tmp_path / "retrieval.json"))

    result = s.search("абракадабра", k=2)

    assert len(result) == 2


def test_embedding_cache_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SentenceTransformer", DummyModel)
    DummyModel.document_encode_calls = 0
    store = _store(tmp_path)
    config = _config(tmp_path / "retrieval.json")

    search.Searcher(store, cache_dir=tmp_path, config_path=config)
    search.Searcher(store, cache_dir=tmp_path, config_path=config)

    assert DummyModel.document_encode_calls == 1
    assert list(tmp_path.glob("*.manifest.json"))


def test_cache_invalidates_when_document_strategy_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SentenceTransformer", DummyModel)
    DummyModel.document_encode_calls = 0
    store = _store(tmp_path)
    config = _config(tmp_path / "retrieval.json")

    first = search.Searcher(store, cache_dir=tmp_path, config_path=config).cache_path()
    monkeypatch.setattr(search, "DOCUMENT_STRATEGY_VERSION", "changed-for-test")
    second = search.Searcher(store, cache_dir=tmp_path, config_path=config).cache_path()

    assert first != second
    assert DummyModel.document_encode_calls == 2


def test_pipeline_imports_and_runs_on_current_python(monkeypatch):
    class FakeSearcher:
        reranker = None

        def __init__(self, store, model_name=None, rerank=False):
            self.store = store

        def search(self, text, k=10, k_rerank=50, extra_queries=()):
            return [{"code": item["code"], "name": item["name"], "score": 1.0} for item in self.store.items[:k]]

        def encode(self, texts, query=True):
            return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("ml.pipeline.Searcher", FakeSearcher)

    result = Pipeline(ktru_path=MOCK_PATH).analyze("стул офисный")

    assert result["input"] == "стул офисный"
    assert "candidates" in result
