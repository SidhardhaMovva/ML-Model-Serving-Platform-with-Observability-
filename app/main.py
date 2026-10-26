"""FastAPI model-serving service.

Endpoints:
  GET  /health   — liveness + whether the model artifact is loaded
  POST /predict  — validated iris classification with probabilities
  GET  /models   — model/version metadata for the loaded artifact
  GET  /metrics  — Prometheus text format (requests, latency, predictions)
"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app import metrics
from app.logging_config import configure_logging, trace_id_var
from app.model import ModelNotLoadedError, ModelStore
from app.schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse

logger = configure_logging()
store = ModelStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    loaded = store.load()
    logger.info(
        "model loaded" if loaded else "model artifact missing; /predict will 503",
        extra={"model_version": store.version if loaded else None},
    )
    yield


app = FastAPI(title="ML Model Serving Platform", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
    token = trace_id_var.set(trace_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration = time.perf_counter() - start
    response.headers["x-trace-id"] = trace_id
    metrics.REQUEST_COUNT.labels(
        method=request.method, path=request.url.path, status=str(response.status_code)
    ).inc()
    metrics.REQUEST_LATENCY.labels(
        method=request.method, path=request.url.path
    ).observe(duration)
    logger.info(
        "request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        },
    )
    trace_id_var.reset(token)
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if store.loaded else "degraded",
        model_loaded=store.loaded,
        model_version=store.version if store.loaded else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        class_id, probabilities = store.predict(payload.as_feature_vector())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    label = store.class_label(class_id)
    metrics.PREDICTION_COUNT.labels(
        model_name=store.name, model_version=store.version, predicted_class=label
    ).inc()
    logger.info(
        "prediction served",
        extra={"model_version": store.version, "prediction": label},
    )
    return PredictResponse(
        prediction=label,
        class_id=class_id,
        probabilities=probabilities,
        model_name=store.name,
        model_version=store.version,
        trace_id=trace_id_var.get(),
    )


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    if not store.loaded:
        return []
    md = store.metadata
    return [
        ModelInfo(
            name=md.get("name", "unknown"),
            version=md.get("version", "unknown"),
            algorithm=md.get("algorithm", "unknown"),
            trained_at=md.get("trained_at"),
            training_accuracy=md.get("training_accuracy"),
            test_accuracy=md.get("test_accuracy"),
            features=md.get("features", []),
            classes=md.get("classes", []),
        )
    ]


@app.get("/metrics")
def prometheus_metrics():
    return PlainTextResponse(
        content=metrics.render_metrics().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
