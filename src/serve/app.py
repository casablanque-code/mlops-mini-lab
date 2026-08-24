"""Инференс-сервис: загружает модель и отдаёт предсказания через REST API."""
import os
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="mlops-mini-lab inference", version="0.1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Модель не найдена по пути {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Вектор признаков, порядок как при обучении")


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
        proba = model.predict_proba([req.features])[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = int(proba.argmax())
    return PredictResponse(prediction=prediction, probability=float(proba[prediction]))
