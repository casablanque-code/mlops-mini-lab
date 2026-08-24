"""Inference service.

Loads a model via an MLflow model URI (registry, S3/GCS, or a local
path) and serves predictions over a REST API, with strict request
validation and Prometheus metrics.
"""
import logging
import math
import os
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, conlist, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mlops-mini-lab.serve")

# Examples:
#   models:/mlops-mini-lab/Production   -> MLflow Model Registry, by stage
#   models:/mlops-mini-lab/3            -> MLflow Model Registry, by version
#   runs:/<run_id>/model                -> a specific run's logged model
#   file:./models/model.joblib          -> plain local file (dev/demo default)
MODEL_URI = os.environ.get("MODEL_URI", "file:./models/model.joblib")
N_FEATURES = int(os.environ.get("N_FEATURES", "8"))

app = FastAPI(title="mlops-mini-lab inference", version="0.2.0")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_model = None
_feature_names: List[str] = []


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
        _model = mlflow.pyfunc.load_model(MODEL_URI)
        # The underlying sklearn estimator exposes feature_names_in_;
        # pyfunc wraps it, so we reach into the raw model when available.
        try:
            raw = _model.unwrap_python_model()
            _feature_names = list(getattr(raw, "feature_names_in_", []))
        except Exception:
            _feature_names = []
        logger.info("Loaded model from MLflow URI %s", MODEL_URI)

    return _model


def get_model():
    if _model is None:
        _load_model()
    return _model


@app.on_event("startup")
def startup():
    try:
        _load_model()
    except Exception as exc:
        # Don't crash the process on startup -- /health will report the
        # problem and /predict will fail loudly, which is easier to debug
        # in an orchestrated environment than a crash-loop.
        logger.error("Model failed to load at startup: %s", exc)


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
