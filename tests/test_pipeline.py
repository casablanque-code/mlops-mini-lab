"""Core tests: data generation, training on a small sample, API health
check, request validation, and metrics endpoint."""
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"Command {cmd} failed:\n{result.stdout}\n{result.stderr}"


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


def _make_test_model(tmp_path, monkeypatch, n_features=5):
    """Trains a tiny model and points MODEL_URI/N_FEATURES at it."""
    raw = tmp_path / "dataset.csv"
    processed = tmp_path / "processed"
    model_out = tmp_path / "model.joblib"

    run([
        sys.executable, "src/data/make_dataset.py",
        "--n-samples", "200", "--n-features", str(n_features),
        "--output", str(raw),
    ])
    run([
        sys.executable, "src/data/prepare.py",
        "--input", str(raw), "--output-dir", str(processed),
    ])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    run([
        sys.executable, "src/train.py",
        "--train-path", str(processed / "train.csv"),
        "--test-path", str(processed / "test.csv"),
        "--n-estimators", "20",
        "--model-out", str(model_out),
    ])
    monkeypatch.setenv("MODEL_URI", f"file:{model_out}")
    monkeypatch.setenv("N_FEATURES", str(n_features))


def _fresh_app():
    """Re-imports the app module so it picks up env vars set by the test."""
    import importlib
    import src.serve.app as app_module

    importlib.reload(app_module)
    return app_module.app


def test_api_health_without_model(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "file:models/does_not_exist.joblib")
    app = _fresh_app()

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "model_missing"


def test_predict_with_valid_features(tmp_path, monkeypatch):
    _make_test_model(tmp_path, monkeypatch, n_features=5)
    app = _fresh_app()
    client = TestClient(app)

    resp = client.post("/predict", json={"features": [0.1, 0.2, -0.3, 0.4, 0.0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_wrong_length(tmp_path, monkeypatch):
    _make_test_model(tmp_path, monkeypatch, n_features=5)
    app = _fresh_app()
    client = TestClient(app)

    resp = client.post("/predict", json={"features": [0.1, 0.2]})
    assert resp.status_code == 422


def test_predict_rejects_nan(tmp_path, monkeypatch):
    _make_test_model(tmp_path, monkeypatch, n_features=5)
    app = _fresh_app()
    client = TestClient(app)

    resp = client.post(
        "/predict", json={"features": [0.1, 0.2, float("nan"), 0.4, 0.0]}
    )
    assert resp.status_code == 422


def test_metrics_endpoint_exposed(tmp_path, monkeypatch):
    _make_test_model(tmp_path, monkeypatch, n_features=5)
    app = _fresh_app()
    client = TestClient(app)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests" in resp.content or b"# HELP" in resp.content
