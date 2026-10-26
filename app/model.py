"""Model loading and inference.

The model artifact is produced by ``training/train.py`` and consists of:
  - ``model.joblib``    — the fitted scikit-learn pipeline
  - ``metadata.json``   — name, version, algorithm, accuracies, feature/class lists

The service refuses to serve predictions if the artifact is missing, and
reports that state honestly via /health instead of fabricating predictions.
"""
import json
import os
from pathlib import Path

import joblib

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelNotLoadedError(RuntimeError):
    pass


class ModelStore:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(
            model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        self._model = None
        self.metadata: dict = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        model_path = self.model_dir / "model.joblib"
        metadata_path = self.model_dir / "metadata.json"
        if not model_path.exists():
            return False
        self._model = joblib.load(model_path)
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
        return True

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    @property
    def name(self) -> str:
        return self.metadata.get("name", "unknown")

    def predict(self, features: list[float]) -> tuple[int, dict[str, float]]:
        if not self.loaded:
            raise ModelNotLoadedError(
                "Model artifact not found. Run `python -m training.train` first."
            )
        probabilities = self._model.predict_proba([features])[0]
        class_id = int(probabilities.argmax())
        classes = self.metadata.get("classes") or list(
            self._model.classes_.astype(str)
        )
        probs = {
            str(classes[i]): round(float(p), 6)
            for i, p in enumerate(probabilities)
        }
        return class_id, probs

    def class_label(self, class_id: int) -> str:
        classes = self.metadata.get("classes", [])
        if classes and 0 <= class_id < len(classes):
            return str(classes[class_id])
        return str(class_id)
