# Technical documentation

## Dataset

[IBM Telco Customer Churn](https://huggingface.co/datasets/aai510-group1/telco-customer-churn)
via Hugging Face (`aai510-group1/telco-customer-churn`), a well-known,
publicly available churn benchmark. Pre-split by the source into:

| Split | Rows |
|---|---|
| Train | 4,225 |
| Validation | 1,409 |
| Test | 1,409 |

Downloaded once and committed as CSVs under `data/` — the app and CI never
need network access or an API key. `src/download_data.py` (see below,
mirroring the pattern in the NLP project) can regenerate them.

## Feature engineering (`src/features.py`)

12 numeric columns (tenure, charges, referrals, satisfaction score, etc.)
and 13 categorical columns (contract type, internet type, payment method,
service add-ons), one-hot encoded to 46 total model features.

**Deliberately excluded columns** — this is the part that matters most for
an honest result: the raw dataset also ships `Churn Score`, `CLTV`,
`Churn Category`, `Churn Reason`, and `Customer Status`. `Churn Score` in
particular is IBM's *own* pre-computed churn-propensity score bundled with
the data — training on it would make the model trivially "predict" a
number that already encodes the answer. None of these five columns are in
`FEATURE_COLS`. `Satisfaction Score` was kept because it's a legitimate,
independently-collected signal (a customer survey response), not a
derivative of the churn label itself.

`features.py` is imported by both `train.py` and `api.py` — this is the
detail that keeps a real MLOps pipeline honest: if the serving-time feature
logic ever diverged from training-time logic, predictions would silently
degrade. Sharing the module means that class of bug is structurally
impossible here.

## Training (`src/train.py`)

- Model: `GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42)`
- Tracked with MLflow against a local SQLite backend (`mlflow.db`) — the
  file-store backend (`./mlruns`) is deprecated in MLflow 3.x and refuses to
  initialize, so this project uses `sqlite:///mlflow.db` instead. Logged per
  run: all five hyperparameters, train-row count, feature count, and all ten
  val/test metrics, plus the serialized model artifact.
- Latest run ID: `c0d33cc8ed794c159a1958e60a451910` (`model/metrics.json`).

### Results

| Split | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| Validation | 96.24% | 92.65% | 96.25% | 89.30% | 0.9929 |
| Test | 95.74% | 91.69% | 95.11% | 88.50% | 0.9926 |

Validation and test metrics track each other closely (no meaningful gap),
which is the honest sanity check that the model isn't overfit to the
validation split specifically.

## Drift monitoring (`src/monitor.py`)

A two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) per numeric
feature, comparing the raw training-time values (saved to
`model/training_baseline.json` at train time) against an incoming batch.
`p < 0.05` on any feature flags that feature as drifted.

**Verified both directions, not just the happy path:**
- Running the monitor against the real, unmodified test split → `drift_detected: false` on all 12 features (expected — it's an honest held-out split from the same population as training).
- Running it against the same split with `Monthly Charge × 1.4` and `Tenure × 0.6` applied → flags exactly those two features (`p = 0.0` on both), the rest stay clean. This is what proves the detector responds to actual distributional change rather than always returning one fixed answer.

The dashboard's "Simulate 40% shift" button calls
`GET /monitoring/drift-demo?shift=0.4`, which applies this same synthetic
shift server-side — clearly labeled as a demo, not real production traffic.

## Serving (`src/api.py`)

FastAPI app with:
- `GET /health` — liveness + which MLflow run is currently loaded
- `POST /predict` — takes a `CustomerFeatures` payload (Pydantic-validated),
  runs it through the identical `build_feature_frame` used at training time,
  reindexes to the training-time column set (`reindex(..., fill_value=0)`
  handles any category not seen at inference), and returns a probability +
  risk tier
- `GET /monitoring/metrics` — the last training run's full metrics.json
- `GET /monitoring/drift-demo` — the KS-test check described above
- `GET /` — a small dashboard (`static/dashboard.html`) rendering the above
  as cards + a live drift table, styled consistently with the rest of this
  portfolio's dark design system

Verified live, not just unit-tested: a month-to-month, 2-month-tenure,
satisfaction-score-1 customer predicts 99.4% churn probability; a two-year
contract customer with 60 months tenure and satisfaction score 5 predicts
0.25% — both directionally exactly what a churn model should say.

## Testing (`tests/test_pipeline.py`)

6 tests, all passing against the real trained model (not mocks):
1. Feature frame has zero nulls after engineering.
2. `/health` returns 200 with `status: ok`.
3. `/predict` returns a probability in `[0, 1]`.
4. `/monitoring/metrics` includes `test_accuracy`.
5. Drift check on the unmodified holdout returns `drift_detected: False`.
6. Drift check on a 2x Monthly-Charge shift returns `drift_detected: True`.

## CI/CD (`.github/workflows/ci.yml`)

Two jobs on every push/PR to `main`:
1. **test** — install deps, `ruff check src tests` (lint), `pytest tests/ -v`.
2. **docker-build** — builds the image from the committed `Dockerfile`
   (depends on `test` passing first).

Both were run locally before being committed: `ruff check` returns
"All checks passed!", `pytest` passes 6/6, and `docker build` +
`docker run` was verified end-to-end (container's `/health` and `/predict`
both respond correctly on a freshly built image) — see the Docker section
below.

## Docker

`Dockerfile` installs `requirements.txt`, copies `src/`, `model/`, and the
test CSV (needed for the `/monitoring/drift-demo` route's demo comparison
batch), and runs `uvicorn api:app --app-dir src`. Built and run-tested
locally: `docker build` succeeds, and a container from that image answers
`/health` with the correct run ID and returns a correct churn probability
from `/predict` — this was checked before writing this doc, not assumed.

## What was deliberately not built

- **No hosted model registry / remote artifact store.** MLflow's local
  SQLite backend is enough to demonstrate real experiment tracking without
  requiring a paid service — consistent with this portfolio's "free hosting
  only" constraint on infra-heavy projects.
- **No automated retraining trigger.** A production system would retrain on
  a drift alert; here the drift check is monitoring-only, and wiring an
  automatic retrain is flagged as the natural next step rather than
  something silently half-implemented.
- **No calibrated probabilities beyond scikit-learn's default
  `predict_proba`.** `GradientBoostingClassifier`'s probabilities are
  reasonably well-calibrated in practice for this model family, but no
  separate `CalibratedClassifierCV` step was added — noted rather than
  silently assumed.
