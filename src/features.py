"""Shared feature engineering so training and serving never drift apart."""
import pandas as pd

NUMERIC_COLS = [
    "Age",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Satisfaction Score",
]

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Contract",
    "Internet Type",
    "Payment Method",
    "Offer",
    "Referred a Friend",
    "Paperless Billing",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Premium Tech Support",
    "Unlimited Data",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "Churn"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Selects + cleans the raw Telco churn columns into a model-ready frame."""
    frame = df[FEATURE_COLS].copy()

    for col in NUMERIC_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_COLS:
        frame[col] = frame[col].astype("string").fillna("Unknown")

    frame = pd.get_dummies(frame, columns=CATEGORICAL_COLS, dummy_na=False)
    return frame


def align_columns(frame: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """Aligns a one-hot-encoded frame to the training-time column set (for serving)."""
    for col in reference_columns:
        if col not in frame.columns:
            frame[col] = 0
    return frame[reference_columns]
