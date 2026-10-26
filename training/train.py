"""Train the iris classifier and save the serving artifact.

Run:
    python -m training.train [--output-dir models] [--version 1.0.0]

Produces ``model.joblib`` and ``metadata.json`` in the output directory.
The dataset is scikit-learn's bundled iris dataset (no download needed),
split with a fixed seed so results are reproducible.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train(output_dir: Path, version: str = "1.0.0") -> dict:
    iris = load_iris()
    x, y = iris.data, iris.target
    classes = [str(name) for name in iris.target_names]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    metadata = {
        "name": "iris-classifier",
        "version": version,
        "algorithm": "RandomForestClassifier(n_estimators=100)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_accuracy": round(
            float(accuracy_score(y_train, model.predict(x_train))), 4
        ),
        "test_accuracy": round(
            float(accuracy_score(y_test, model.predict(x_test))), 4
        ),
        "features": FEATURES,
        "classes": classes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
    )
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    metadata = train(args.output_dir, args.version)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
