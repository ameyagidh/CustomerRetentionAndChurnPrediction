import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api
from features import build_feature_frame
from monitor import check_drift

client = TestClient(api.app)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_feature_frame_has_no_nulls():
    df = pd.read_csv(os.path.join(BASE, "data", "churn_train.csv"))
    frame = build_feature_frame(df)
    assert frame.isnull().sum().sum() == 0
    assert len(frame) == len(df)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    resp = client.post("/predict", json={"Contract": "Two Year", "Tenure_in_Months": 60})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_monitoring_metrics_endpoint():
    resp = client.get("/monitoring/metrics")
    assert resp.status_code == 200
    assert "test_accuracy" in resp.json()["metrics"]


def test_drift_not_flagged_on_unmodified_holdout():
    prod_df = pd.read_csv(os.path.join(BASE, "data", "churn_test.csv"))
    report = check_drift(prod_df)
    assert report["drift_detected"] is False


def test_drift_flagged_on_synthetic_shift():
    prod_df = pd.read_csv(os.path.join(BASE, "data", "churn_test.csv")).copy()
    prod_df["Monthly Charge"] = prod_df["Monthly Charge"].astype(float) * 2.0
    report = check_drift(prod_df)
    assert report["drift_detected"] is True
