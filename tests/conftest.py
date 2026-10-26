"""Train a real model artifact into a temp dir and point the app at it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import train  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def trained_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    train(model_dir, version="test-1.0.0")
    return model_dir


@pytest.fixture()
def client(trained_model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    # Rebuild the store/app against the test artifact.
    import importlib

    import app.main as main_module

    importlib.reload(main_module)
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        yield test_client
