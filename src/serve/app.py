"""Inference service.

Loads a model via an MLflow model URI (registry, S3/GCS, or a local
path) and serves predictions over a REST API, with strict request
validation and Prometheus metrics.
"""
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, conlist, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mlops-mini-lab.serve")

# Examples:
#   models:/mlops-mini-lab@production   -> MLflow Model Registry, by alias
#     (MLflow >= 2.9: stages like /Production are deprecated in favor of
#      aliases -- see docs/runbook.md for how to set one)
#   models:/mlops-mini-lab/3            -> MLflow Model Registry, by version
#   runs:/<run_id>/model                -> a specific run's logged model
#   file:./models/model.joblib          -> plain local file (dev/demo default)
MODEL_URI = os.environ.get("MODEL_URI", "file:./models/model.joblib")
N_FEATURES = int(os.environ.get("N_FEATURES", "8"))

_model = None
_feature_names: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except Exception as exc:
        # Don't crash the process on startup -- /health will report the
        # problem and /predict will fail loudly, which is easier to debug
        # in an orchestrated environment than a crash-loop.
        logger.error("Model failed to load at startup: %s", exc)
    yield


app = FastAPI(title="mlops-mini-lab inference", version="0.2.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request with method, path, status code, and latency.

    This is deliberately basic (structured enough to grep, not a full
    structured-logging setup) -- it's the minimum needed to answer
    "what happened and how long did it take" without a metrics backend,
    and it's the natural precursor to shipping these fields to a log
    aggregator later.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def _sanitize_for_json(obj):
    """Replaces NaN/Inf floats with None so error payloads containing the
    offending input (e.g. Pydantic's validation error 'input' field) can
    still be JSON-serialized -- Starlette's JSONResponse rejects NaN/Inf
    outright (allow_nan=False)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _load_model():
    """Loads the model from MODEL_URI. Supports the MLflow Model Registry,
    a specific run, or a plain local joblib file via the file: scheme."""
    global _model, _feature_names

    if MODEL_URI.startswith("file:"):
        import joblib

        local_path = MODEL_URI.removeprefix("file:")
        if not os.path.exists(local_path):
            raise RuntimeError(f"Model file not found at {local_path}")
        _model = joblib.load(local_path)
        _feature_names = list(getattr(_model, "feature_names_in_", []))
        logger.info("Loaded local model from %s", local_path)
    else:
        # Use the sklearn flavor loader (not mlflow.pyfunc) so we get a
        # real RandomForestClassifier back, with predict_proba intact.
        # mlflow.pyfunc.load_model() returns a generic wrapper that only
        # exposes predict() -- fine for a class label, but it silently
        # drops probabilities, which this service's /predict response
        # relies on.
        _model = mlflow.sklearn.load_model(MODEL_URI)
        _feature_names = list(getattr(_model, "feature_names_in_", []))
        logger.info(
            "Loaded model from MLflow URI %s (feature_names=%s)",
            MODEL_URI,
            _feature_names,
        )

    return _model


def get_model():
    if _model is None:
        _load_model()
    return _model


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Default handler serializes exc.errors() with plain json.dumps
    (allow_nan=False), which raises ValueError whenever the invalid input
    itself was NaN/Inf -- sanitize before returning."""
    payload = _sanitize_for_json(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": payload})


class PredictRequest(BaseModel):
    features: conlist(float, min_length=N_FEATURES, max_length=N_FEATURES) = Field(
        ..., description=f"Feature vector, exactly {N_FEATURES} values, same order as training"
    )

    @field_validator("features")
    @classmethod
    def no_nan_or_inf(cls, v: List[float]) -> List[float]:
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("features must not contain NaN or Inf")
        return v


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    ok = _model is not None
    return {
        "status": "ok" if ok else "model_missing",
        "model_uri": MODEL_URI,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    try:
        if _feature_names:
            X = pd.DataFrame([req.features], columns=_feature_names)
        else:
            X = pd.DataFrame([req.features])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            prediction = int(proba.argmax())
            probability = float(proba[prediction])
        else:
            # mlflow.pyfunc models expose predict() only; fall back to it.
            result = model.predict(X)
            prediction = int(result[0])
            probability = 1.0
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(prediction=prediction, probability=probability)
