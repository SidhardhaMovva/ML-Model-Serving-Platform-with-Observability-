# ML Model Serving Platform with Observability

A small, working model-serving service: a scikit-learn classifier trained by a
reproducible script, served over FastAPI, with Prometheus metrics and
structured JSON logging built in from the start.

The point of this project is the lifecycle around the model — train → save →
load → serve → observe — implemented plainly and covered by real tests,
rather than a long feature list.

## What it does

- **Trains a real model.** `training/train.py` trains a
  `RandomForestClassifier` on scikit-learn's bundled iris dataset (fixed seed,
  stratified 80/20 split) and saves `model.joblib` + `metadata.json`
  (name, version, algorithm, train/test accuracy, features, classes).
- **Serves predictions** through FastAPI with validated request/response
  schemas. Responses include class probabilities, the model name and version,
  and a per-request trace ID.
- **Exposes observability endpoints:**
  - `/metrics` — Prometheus text format: HTTP request counts and latency
    histograms per route, and a prediction counter labelled by model name,
    version, and predicted class. Counters are tracked in code
    (`app/metrics.py`), not mocked.
  - Structured JSON logs — one JSON object per line, with the request's
    `trace_id` on every record so logs for one request can be correlated.
- **Reports its state honestly.** If the model artifact is missing, `/health`
  returns `degraded` and `/predict` returns 503 — it never fabricates a
  prediction.

## Architecture

```
training/train.py ──► models/model.joblib + metadata.json
                                │
                     app/model.py (ModelStore: load / predict)
                                │
   client ──► app/main.py (FastAPI)
                ├─ middleware: trace ID, JSON logging, request/latency metrics
                ├─ POST /predict   (validated schemas, prediction metric)
                ├─ GET  /models    (artifact metadata)
                ├─ GET  /health    (liveness + model state)
                └─ GET  /metrics   (Prometheus text)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m training.train          # creates models/model.joblib + metadata.json
uvicorn app.main:app --port 8000
```

Example:

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"model_version":"1.0.0"}

curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# {"prediction":"setosa","class_id":0,
#  "probabilities":{"setosa":1.0,"versicolor":0.0,"virginica":0.0},
#  "model_name":"iris-classifier","model_version":"1.0.0","trace_id":"…"}

curl -s localhost:8000/metrics | grep model_predictions_total
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite (9 tests) trains a real model into a temporary directory and
exercises the API against it: prediction correctness for known iris samples,
schema validation failures (422), health/model metadata, metrics incrementing
after a prediction, and trace-ID propagation — plus unit tests for the
model store's load/predict lifecycle and its missing-artifact behaviour.

## Docker

```bash
docker compose up --build   # trains the model at image build, serves on :8000
```

## Limitations (honest ones)

- One model, loaded at startup. No hot-swapping, A/B routing, or model
  registry — `/models` reports the single loaded artifact's metadata.
- In-process metrics only; restarting the process resets the counters. For
  production you'd scrape `/metrics` with Prometheus and aggregate across
  replicas.
- The iris model is a teaching-scale example (held-out test accuracy 0.90
  with the committed training configuration). The serving/observability code
  is model-agnostic, but nothing here has been load-tested.
- No authentication on any endpoint.

## License

MIT — see `LICENSE`.
