"""
FastAPI serving layer for the churn model.

Endpoints:
  GET  /health              - liveness + model metadata
  POST /predict              - single-customer churn prediction
  GET  /monitoring/metrics   - the training/validation/test metrics from the last run
  POST /monitoring/drift     - KS-test drift check against a batch of feature rows
"""
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_feature_frame
from monitor import check_drift

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Telco Churn MLOps API", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_bundle = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
_model = _bundle["model"]
_feature_columns = _bundle["feature_columns"]

import json

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    _metrics = json.load(f)


class CustomerFeatures(BaseModel):
    Age: int = 45
    Number_of_Dependents: int = 0
    Number_of_Referrals: int = 1
    Tenure_in_Months: int = 12
    Avg_Monthly_Long_Distance_Charges: float = 15.0
    Avg_Monthly_GB_Download: float = 10.0
    Monthly_Charge: float = 70.0
    Total_Charges: float = 840.0
    Total_Refunds: float = 0.0
    Total_Extra_Data_Charges: float = 0.0
    Total_Long_Distance_Charges: float = 180.0
    Satisfaction_Score: int = 3
    Gender: str = "Male"
    Married: str = "0"
    Contract: str = "Month-to-Month"
    Internet_Type: str = "Fiber Optic"
    Payment_Method: str = "Bank Withdrawal"
    Offer: str | None = None
    Referred_a_Friend: str = "0"
    Paperless_Billing: str = "1"
    Multiple_Lines: str = "0"
    Online_Security: str = "0"
    Online_Backup: str = "0"
    Premium_Tech_Support: str = "0"
    Unlimited_Data: str = "1"


_FIELD_TO_COLUMN = {
    "Number_of_Dependents": "Number of Dependents",
    "Number_of_Referrals": "Number of Referrals",
    "Tenure_in_Months": "Tenure in Months",
    "Avg_Monthly_Long_Distance_Charges": "Avg Monthly Long Distance Charges",
    "Avg_Monthly_GB_Download": "Avg Monthly GB Download",
    "Monthly_Charge": "Monthly Charge",
    "Total_Charges": "Total Charges",
    "Total_Refunds": "Total Refunds",
    "Total_Extra_Data_Charges": "Total Extra Data Charges",
    "Total_Long_Distance_Charges": "Total Long Distance Charges",
    "Satisfaction_Score": "Satisfaction Score",
    "Internet_Type": "Internet Type",
    "Payment_Method": "Payment Method",
    "Referred_a_Friend": "Referred a Friend",
    "Paperless_Billing": "Paperless Billing",
    "Multiple_Lines": "Multiple Lines",
    "Online_Security": "Online Security",
    "Online_Backup": "Online Backup",
    "Premium_Tech_Support": "Premium Tech Support",
    "Unlimited_Data": "Unlimited Data",
}


def _to_row(payload: CustomerFeatures) -> pd.DataFrame:
    data = payload.model_dump()
    row = {_FIELD_TO_COLUMN.get(k, k): v for k, v in data.items()}
    return pd.DataFrame([row])


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": _metrics["run_id"], "num_features": len(_feature_columns)}


@app.post("/predict")
def predict(payload: CustomerFeatures):
    row = _to_row(payload)
    features = build_feature_frame(row).reindex(columns=_feature_columns, fill_value=0)
    proba = float(_model.predict_proba(features)[0, 1])
    prediction = int(proba >= 0.5)
    risk = "high" if proba >= 0.66 else "medium" if proba >= 0.33 else "low"
    return {"churn_prediction": prediction, "churn_probability": round(proba, 4), "risk_tier": risk}


@app.get("/monitoring/metrics")
def monitoring_metrics():
    return _metrics


@app.get("/monitoring/drift-demo")
def monitoring_drift_demo(shift: float = 0.0):
    """
    Demo endpoint: re-runs the KS drift test using the real held-out test split,
    optionally with a synthetic additive shift applied to Monthly Charge and
    Tenure to demonstrate what a genuine distribution shift trips the alert on.
    shift=0 uses the unmodified real test data (no drift expected).
    """
    prod_df = pd.read_csv(os.path.join(BASE, "data", "churn_test.csv")).copy()
    if shift:
        prod_df["Monthly Charge"] = prod_df["Monthly Charge"].astype(float) * (1 + shift)
        prod_df["Tenure in Months"] = prod_df["Tenure in Months"].astype(float) * (1 - shift)
    report = check_drift(prod_df)
    report["simulated_shift_applied"] = shift
    return report


@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join(STATIC_DIR, "dashboard.html")) as f:
        return f.read()
