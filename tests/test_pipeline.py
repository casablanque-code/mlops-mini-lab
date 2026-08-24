"""Базовые тесты: генерация данных, обучение на выборке, работоспособность API."""
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"Команда {cmd} упала:\n{result.stdout}\n{result.stderr}"


def test_make_dataset(tmp_path):
    out = tmp_path / "dataset.csv"
    run([
        sys.executable, "src/data/make_dataset.py",
        "--n-samples", "200", "--n-features", "5",
        "--output", str(out),
    ])
    df = pd.read_csv(out)
    assert len(df) == 200
    assert "target" in df.columns


def test_prepare(tmp_path):
    raw = tmp_path / "dataset.csv"
    run([
        sys.executable, "src/data/make_dataset.py",
        "--n-samples", "200", "--n-features", "5",
        "--output", str(raw),
    ])
    out_dir = tmp_path / "processed"
    run([
        sys.executable, "src/data/prepare.py",
        "--input", str(raw), "--output-dir", str(out_dir),
    ])
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()


def test_train_smoke(tmp_path, monkeypatch):
    raw = tmp_path / "dataset.csv"
    processed = tmp_path / "processed"
    model_out = tmp_path / "model.joblib"

    run([sys.executable, "src/data/make_dataset.py", "--n-samples", "200", "--n-features", "5", "--output", str(raw)])
    run([sys.executable, "src/data/prepare.py", "--input", str(raw), "--output-dir", str(processed)])

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    run([
        sys.executable, "src/train.py",
        "--train-path", str(processed / "train.csv"),
        "--test-path", str(processed / "test.csv"),
        "--n-estimators", "20",
        "--model-out", str(model_out),
    ])
    assert model_out.exists()
    model = joblib.load(model_out)
    assert hasattr(model, "predict")


def test_api_health_without_model(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "models/does_not_exist.joblib")
    from src.serve.app import app  # импорт после установки переменной окружения

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "model_missing"}
