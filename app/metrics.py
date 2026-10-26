"""Prometheus metrics, tracked in code and exposed at /metrics.

Uses a dedicated registry so the metrics can be rendered without depending
on the default global registry's process/runtime collectors.
"""
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry,
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions served",
    ["model_name", "model_version", "predicted_class"],
    registry=registry,
)


def render_metrics() -> bytes:
    return generate_latest(registry)
