"""
Data-drift monitor: runs a two-sample Kolmogorov-Smirnov test between the
training-time distribution of each numeric feature and a batch of incoming
"production" data, so a genuine distribution shift (not just a vibe) is what
trips the alert.
"""
import json
import os

import pandas as pd
from scipy.stats import ks_2samp

from features import NUMERIC_COLS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")

DRIFT_P_THRESHOLD = 0.05


def load_baseline():
    with open(os.path.join(MODEL_DIR, "training_baseline.json")) as f:
        return json.load(f)


def check_drift(production_df: pd.DataFrame) -> dict:
    baseline = load_baseline()
    results = []
    for col in NUMERIC_COLS:
        if col not in production_df.columns:
            continue
        baseline_values = baseline[col]
        prod_values = pd.to_numeric(production_df[col], errors="coerce").dropna().tolist()
        if not prod_values:
            continue
        stat, p_value = ks_2samp(baseline_values, prod_values)
        results.append(
            {
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drifted": bool(p_value < DRIFT_P_THRESHOLD),
            }
        )
    any_drift = any(r["drifted"] for r in results)
    return {"drift_detected": any_drift, "threshold_p_value": DRIFT_P_THRESHOLD, "features": results}


def main():
    # Demo: treat the held-out test split as the "production" batch to monitor.
    prod_df = pd.read_csv(os.path.join(BASE, "data", "churn_test.csv"))
    report = check_drift(prod_df)

    with open(os.path.join(MODEL_DIR, "drift_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Drift detected: {report['drift_detected']}")
    for r in report["features"]:
        flag = "DRIFTED" if r["drifted"] else "ok"
        print(f"  {r['feature']:35s} p={r['p_value']:.4f}  [{flag}]")


if __name__ == "__main__":
    main()
