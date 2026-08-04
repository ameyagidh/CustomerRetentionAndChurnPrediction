"""
Regenerates data/churn_*.csv from the original Hugging Face dataset.
Not required to run the app - the CSVs are already committed - but kept
for reproducibility. Requires requirements-datasets.txt.
"""
import os

import pandas as pd
from datasets import load_dataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def main():
    for split in ["train", "validation", "test"]:
        d = load_dataset("aai510-group1/telco-customer-churn", split=split)
        pd.DataFrame(d).to_csv(os.path.join(DATA, f"churn_{split}.csv"), index=False)
        print(f"{split}: refreshed")


if __name__ == "__main__":
    main()
