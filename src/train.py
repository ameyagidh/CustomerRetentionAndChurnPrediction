"""
Trains a churn-prediction model on the bundled Telco Customer Churn data,
tracking every run (params, metrics, model artifact) in a local MLflow
tracking store under mlruns/ - no cloud, no API key.
"""
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import NUMERIC_COLS, TARGET_COL, build_feature_frame

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
mlflow.set_experiment("telco-churn")


def load_split(name):
    return pd.read_csv(os.path.join(DATA, f"churn_{name}.csv"))


def main():
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    X_train = build_feature_frame(train_df)
    y_train = train_df[TARGET_COL]

    X_val = build_feature_frame(val_df).reindex(columns=X_train.columns, fill_value=0)
    y_val = val_df[TARGET_COL]

    X_test = build_feature_frame(test_df).reindex(columns=X_train.columns, fill_value=0)
    y_test = test_df[TARGET_COL]

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="gbc-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("num_features", X_train.shape[1])

        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        val_preds = clf.predict(X_val)
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, val_preds),
            "val_f1": f1_score(y_val, val_preds),
            "val_precision": precision_score(y_val, val_preds),
            "val_recall": recall_score(y_val, val_preds),
            "val_auc": roc_auc_score(y_val, val_proba),
        }

        test_preds = clf.predict(X_test)
        test_proba = clf.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, test_preds),
            "test_f1": f1_score(y_test, test_preds),
            "test_precision": precision_score(y_test, test_preds),
            "test_recall": recall_score(y_test, test_preds),
            "test_auc": roc_auc_score(y_test, test_proba),
        }

        all_metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)
        mlflow.sklearn.log_model(clf, "model")

        print(f"MLflow run: {run.info.run_id}")
        for k, v in all_metrics.items():
            print(f"  {k}: {v:.4f}")

        joblib.dump(
            {"model": clf, "feature_columns": list(X_train.columns)},
            os.path.join(MODEL_DIR, "churn_model.joblib"),
        )

        with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
            json.dump(
                {
                    "run_id": run.info.run_id,
                    "params": params,
                    "metrics": all_metrics,
                    "train_rows": len(X_train),
                    "val_rows": len(X_val),
                    "test_rows": len(X_test),
                    "num_features": X_train.shape[1],
                },
                f,
                indent=2,
            )

        # Baseline distributions of the raw numeric features (pre-encoding),
        # used by src/monitor.py to run a KS-test against incoming production data.
        baseline_stats = {
            col: pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).tolist()
            for col in NUMERIC_COLS
        }
        with open(os.path.join(MODEL_DIR, "training_baseline.json"), "w") as f:
            json.dump(baseline_stats, f)

    print("\nSaved model/churn_model.joblib, model/metrics.json, model/training_baseline.json")


if __name__ == "__main__":
    main()
