"""End-to-end API tests against a real trained model."""

SETOSA = {  # a classic setosa measurement from the iris dataset
    "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2,
}
VIRGINICA = {
    "sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3,
}


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1.0.0"


def test_predict_setosa(client):
    resp = client.post("/predict", json=SETOSA)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "setosa"
    assert body["class_id"] == 0
    assert body["model_version"] == "test-1.0.0"
    assert body["probabilities"]["setosa"] > 0.5
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
    assert body["trace_id"] != "-"


def test_predict_virginica(client):
    resp = client.post("/predict", json=VIRGINICA)
    assert resp.status_code == 200
    assert resp.json()["prediction"] == "virginica"


def test_predict_rejects_invalid_input(client):
    # negative measurement violates schema (gt=0)
    resp = client.post("/predict", json={**SETOSA, "sepal_length": -1})
    assert resp.status_code == 422
    # missing field
    resp = client.post("/predict", json={"sepal_length": 5.1})
    assert resp.status_code == 422


def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 1
    info = models[0]
    assert info["name"] == "iris-classifier"
    assert info["version"] == "test-1.0.0"
    assert info["test_accuracy"] is not None and info["test_accuracy"] > 0.8
    assert info["classes"] == ["setosa", "versicolor", "virginica"]


def test_metrics_increment_after_prediction(client):
    client.post("/predict", json=SETOSA)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "http_requests_total" in text
    assert "http_request_duration_seconds" in text
    assert "model_predictions_total" in text
    assert 'predicted_class="setosa"' in text
    assert 'model_version="test-1.0.0"' in text


def test_trace_id_header_is_returned(client):
    resp = client.get("/health", headers={"x-trace-id": "my-trace-123"})
    assert resp.headers["x-trace-id"] == "my-trace-123"
