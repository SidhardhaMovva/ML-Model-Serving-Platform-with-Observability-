"""Unit tests for the model store's load/predict lifecycle."""
from pathlib import Path

import pytest

from app.model import ModelNotLoadedError, ModelStore
from training.train import train


def test_store_predicts_after_training(tmp_path: Path):
    train(tmp_path, version="unit-2.0.0")
    store = ModelStore(model_dir=tmp_path)
    assert store.load() is True
    assert store.version == "unit-2.0.0"
    class_id, probs = store.predict([5.1, 3.5, 1.4, 0.2])
    assert store.class_label(class_id) == "setosa"
    assert probs["setosa"] > 0.5


def test_store_reports_missing_artifact_honestly(tmp_path: Path):
    store = ModelStore(model_dir=tmp_path)
    assert store.load() is False
    assert store.loaded is False
    with pytest.raises(ModelNotLoadedError):
        store.predict([5.1, 3.5, 1.4, 0.2])
