"""Trains a model, logs the run to MLflow, and (optionally) registers it
in the MLflow Model Registry so the serving layer can pull it by
name/stage instead of relying on a local file."""
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
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=None,
        help="If set, registers the trained model in the MLflow Model "
        "Registry under this name (e.g. mlops-mini-lab).",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    with mlflow.start_run() as run:
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
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train.head(2),
            registered_model_name=args.registered_model_name,
        )

        model_out = Path(args.model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        mlflow.log_artifact(str(model_out))

        print("Metrics:", metrics)
        print(f"Model saved locally at: {model_out}")
        print(f"MLflow run id: {run.info.run_id}")
        if args.registered_model_name:
            print(
                f"Registered as '{args.registered_model_name}' (version will be "
                f"shown above/in the MLflow UI).\n"
                f"MLflow stages (Staging/Production) are deprecated -- use an "
                f"alias instead. The CLI command for this may not exist in "
                f"your MLflow version, so use the Python client:\n"
                f"  python -c \"\n"
                f"  from mlflow import MlflowClient\n"
                f"  client = MlflowClient(tracking_uri='{tracking_uri}')\n"
                f"  client.set_registered_model_alias("
                f"'{args.registered_model_name}', 'production', '<N>')  "
                f"# version as a string\n"
                f"  \"\n"
                f"Then point the serving service at:\n"
                f"  MODEL_URI=models:/{args.registered_model_name}@production"
            )


if __name__ == "__main__":
    main()
