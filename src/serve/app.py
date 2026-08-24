"""Inference service: loads the model and serves predictions over a REST API."""
import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="mlops-mini-lab inference", version="0.1.0")

_model = None
_feature_names = None


def get_model():
    global _model, _feature_names
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        _feature_names = list(getattr(_model, "feature_names_in_", []))
    return _model


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector, same order as training")


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    status = "ok" if Path(MODEL_PATH).exists() else "model_missing"
    return {"status": status, "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    try:
        if _feature_names:
            X = pd.DataFrame([req.features], columns=_feature_names)
        else:
            X = [req.features]
        proba = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = int(proba.argmax())
    return PredictResponse(prediction=prediction, probability=float(proba[prediction]))
