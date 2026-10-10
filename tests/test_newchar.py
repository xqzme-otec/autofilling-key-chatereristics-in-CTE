"""New characteristic validity classifier (ml/newchar.py) with a fake encoder: no model download, no data files."""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml.fit import ProductFit
from ml.newchar import INVALID, UNCERTAIN, VALID, VERSION, Features, NewCharClassifier, thresholds


class Encoder:
    """SentenceTransformer.encode shape: a deterministic unit vector per text."""

    def encode(self, texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False):
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha256(t.encode()).digest()[:16], dtype=np.uint8).astype(np.float32) - 127.5
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class Store:
    items = [{"code": "1", "name": "Молоток", "category": "инструменты",
              "characteristics": [{"name": "Масса молотка", "unit": "Килограмм", "values": []}]},
             {"code": "2", "name": "Стол", "category": "мебель",
              "characteristics": [{"name": "Высота", "unit": "Миллиметр", "values": []}]}]


ITEMS = [{"category": "инструменты", "product": "молоток", "name": "Противоскользящая рукоятка", "value": "Да"},
         {"category": "мебель", "product": "стол", "name": "Разрешение экрана", "value": "4K"}]


@pytest.fixture
def model_file(tmp_path):
    portal = tmp_path / "absent.json"
    features = Features(Encoder(), Store(), {}, ProductFit(Store(), portal))
    x = features(ITEMS * 4)
    model = LogisticRegression().fit(x, [1, 0] * 4)
    path = tmp_path / "newchar_model.pkl"
    path.write_bytes(pickle.dumps({"version": VERSION, "model": model, "lo": 0.3, "hi": 0.7}))
    return path, portal


def test_predict_gives_a_zone_and_probability(model_file):
    path, portal = model_file
    clf = NewCharClassifier(Encoder(), Store(), path, portal)
    (zone_ok, p_ok), (zone_bad, p_bad) = clf.predict(ITEMS)
    assert (zone_ok, zone_bad) == (VALID, INVALID) and p_ok > 0.7 > 0.3 > p_bad
    assert clf.predict([]) == []
    assert [clf.zone(p) for p in (0.3, 0.5, 0.7)] == [INVALID, UNCERTAIN, VALID]


def test_model_of_another_version_is_rejected(model_file, tmp_path):
    path, portal = model_file
    saved = pickle.loads(path.read_bytes())
    path.write_bytes(pickle.dumps({**saved, "version": VERSION + 1}))
    with pytest.raises(ValueError, match="rebuild"):
        NewCharClassifier(Encoder(), Store(), path, portal)


def test_thresholds_give_the_widest_zones_with_the_target_precision():
    scores = np.array([0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
    assert thresholds(scores, labels, target=1.0) == (0.2, 0.6)
