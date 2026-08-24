"""Обучение модели с трекингом эксперимента в MLflow."""
import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="data/processed/train.csv")
    parser.add_argument("--test-path", type=str, default="data/processed/test.csv")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--model-out", type=str, default="models/model.joblib")
    parser.add_argument("--experiment", type=str, default="mlops-mini-lab")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("n_features", len(feature_cols))

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_out = Path(args.model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        mlflow.log_artifact(str(model_out))

        print("Метрики:", metrics)
        print(f"Модель сохранена локально: {model_out}")


if __name__ == "__main__":
    main()
