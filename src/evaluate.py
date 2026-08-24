"""Evaluates a saved model on the test set and writes metrics.json (used by dvc metrics)."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/model.joblib")
    parser.add_argument("--test-path", type=str, default="data/processed/test.csv")
    parser.add_argument("--output", type=str, default="metrics.json")
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    test_df = pd.read_csv(args.test_path)
    feature_cols = [c for c in test_df.columns if c != "target"]

    X_test, y_test = test_df[feature_cols], test_df["target"]
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
