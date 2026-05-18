# Technical Documentation

## 1. System Overview

The Insider Threat Detection (ITD) prototype is a web-based system that classifies employee behavioural records as either malicious or benign. It is designed to support security analysts by surfacing high-risk individuals, explaining the model's reasoning through SHAP-based feature contributions, and providing aggregate evaluation metrics across multiple models.

The system consists of three main components:

**Frontend** — A Next.js dashboard where analysts select a model, enter feature values manually or load them from a CSV file, and view threat assessments with SHAP explanations. It also provides a scenario testing panel that runs a sampled batch evaluation against the backend and displays per-model confusion matrices.

**Backend** — A FastAPI service that loads trained model artefacts, validates and vectorises incoming feature data, generates malicious probability scores, applies saved decision thresholds, and computes SHAP contributions for explainability. It also exposes model metrics and the scenario evaluation endpoint.

**ML pipeline** — An offline training pipeline of Python scripts that preprocesses the dataset, tunes hyperparameters with Optuna, trains three classifiers (XGBoost, LightGBM, Random Forest), selects a decision threshold targeting 99% recall, and saves each trained model as a self-contained `.joblib` artefact.

The dataset is a cleaned insider-threat CSV with 118,614 records and a 5.38% positive (malicious) rate. The 21 input features cover employee demographics, access behaviour, printing activity, file-burning activity, travel records, and physical entry patterns.

---

## 2. Software Requirements and Dependencies

**Runtime requirements:**

| Component | Requirement |
|---|---|
| Python | 3.10 or later |
| Node.js | 18 or later |
| npm | Bundled with Node.js |
| Docker + Docker Compose | Optional, for containerised deployment |

**Backend dependencies** (`backend/requirements.txt`):

| Package | Version constraint |
|---|---|
| fastapi | ≥ 0.115.0 |
| uvicorn[standard] | ≥ 0.30.0 |
| joblib | ≥ 1.4.0 |
| scikit-learn | == 1.7.1 |
| xgboost | ≥ 2.0 |
| lightgbm | ≥ 4.0 |
| shap | ≥ 0.46 |

**ML pipeline dependencies** (`ml/requirements.txt`):

| Package | Version constraint |
|---|---|
| scikit-learn | == 1.7.1 |
| xgboost | ≥ 2.0 |
| lightgbm | ≥ 4.0 |
| shap | ≥ 0.46 |
| matplotlib | ≥ 3.7 |
| optuna | ≥ 3.0 |

**Frontend dependencies** (`frontend/itd/package.json`):

| Package | Version |
|---|---|
| next | 16.2.6 |
| react | 19.2.4 |
| react-dom | 19.2.4 |
| axios | ^1.16.1 |
| tailwindcss | ^4 |
| typescript | ^5 |
| eslint | ^9 |

---

## 3. Installation Instructions

### Local installation

Create and activate a Python virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
python -m pip install --upgrade pip
```

Install the backend and ML dependencies:

```bash
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
```

Install frontend dependencies:

```bash
cd frontend/itd
npm install
cd ../..
```

The trained model artefacts must exist in `artifacts/models/` before starting the backend. If they are missing, retrain the models as described in Section 4.

### Docker installation

Docker Compose handles all dependency installation inside the containers:

```bash
docker compose up --build
```

No manual dependency installation is required when using Docker.

---

## 4. Deployment and Execution Instructions

### Running locally

Start the backend from the repository root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend in a separate terminal:

```bash
cd frontend/itd
npm run dev
```

Open the dashboard at `http://localhost:3000`. The backend API is available at `http://localhost:8000`.

### Running with Docker Compose

```bash
docker compose up
```

The backend starts on port `8000` and the frontend on port `3000`. Model artefacts are mounted from `artifacts/models/` and data from `ml/data/`.

### Backend API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/model/status` | Status of the default model |
| GET | `/models/status` | Status of all three models |
| GET | `/metrics/{model_name}` | Saved test-set metrics for a model |
| POST | `/predict` | Predict using the default model |
| POST | `/predict/{model_name}` | Predict using a named model |
| GET | `/scenario-tests/summary` | Batch evaluation over a dataset sample |

Supported `model_name` values: `xgboost`, `lightgbm`, `random_forest`.

### Training the models

Run the following from the `ml/` directory. Hyperparameter tuning is optional if `reports/best_params.json` already exists.

```bash
cd ml

# Optional: re-run Bayesian hyperparameter search (50 Optuna trials per model)
python tune_hyperparameters.py

# Train each model (--target-recall sets the recall threshold, default 0.99)
python train_xgboost.py --target-recall 0.99
python train_lightgbm.py --target-recall 0.99
python train_random_forest.py --target-recall 0.99
```

Trained artefacts are saved to both `ml/models/` and `artifacts/models/`.

### Scenario testing

Run a seeded batch evaluation from the repository root:

```bash
python ml/test_scenarios.py --all-models --n 200 --seed 42 --output ml/reports/testing.md
```

Options:

| Flag | Description |
|---|---|
| `--model` | Single model to evaluate (`xgboost`, `lightgbm`, `random_forest`) |
| `--all-models` | Evaluate all three models on the same sample |
| `--n` | Number of records to sample (default: 50) |
| `--seed` | Random seed for reproducibility (default: 42) |
| `--no-stratify` | Pure random sampling instead of 25% malicious stratification |
| `--output FILE` | Export results to `.md` or `.csv` |

### SHAP and feature importance (optional)

```bash
cd ml
python shap_analysis.py --model xgboost
python feature_importance.py --model models/xgboost_model.joblib --repeats 5
```

---

## 5. Source Code Documentation

### Repository layout

```
.
├── artifacts/
│   └── models/              # Model artefacts used by the backend
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt
├── frontend/
│   └── itd/
│       ├── app/
│       │   ├── page.tsx                        # Landing / login page
│       │   └── (main)/dashboard/page.tsx       # Main analyst dashboard
│       └── services/
│           └── api.ts                          # API client and TypeScript types
├── ml/
│   ├── data/                # Training dataset
│   ├── models/              # Local model artefacts (training output)
│   ├── reports/             # Metrics JSON, threshold sweeps, scenario results
│   ├── training_common.py   # Shared training utilities
│   ├── tune_hyperparameters.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_random_forest.py
│   ├── test_scenarios.py
│   ├── shap_analysis.py
│   └── feature_importance.py
└── docker-compose.yml
```

### Backend — `backend/main.py`

The FastAPI application. On startup it lazily loads model artefacts when the first request is made. Key responsibilities:

- **Feature parsing** — `parse_feature_value` converts raw string inputs to the correct Python type (numeric or categorical), rejecting bad values with a 422 response.
- **Vectorisation** — the saved `DictVectorizer` from training is applied to ensure the same one-hot encoding used during training is reproduced at inference time.
- **Prediction** — `run_prediction` calls `model.predict_proba`, applies the saved `decision_threshold`, and returns a binary label alongside the raw probability.
- **SHAP explanations** — `explain_prediction` dispatches to the appropriate contribution method: `pred_contribs` for XGBoost, `pred_contrib` for LightGBM, and `TreeExplainer` (via SHAP) for Random Forest. Encoded feature contributions are aggregated back to raw feature names before being returned.
- **Scenario evaluation** — `run_scenario_summary` iterates over a sampled batch of dataset records and accumulates TP, TN, FP, and FN counts per model.

### Frontend — `frontend/itd/app/(main)/dashboard/page.tsx`

The main React client component. Key responsibilities:

- **Model selection** — three model cards show live accuracy, recall, precision, and F1 fetched from `/metrics/{model_name}`. Clicking a card switches the active model.
- **Feature input** — a spreadsheet-style grid accepts manual entry. A CSV textarea and file-upload input parse the first data row of a CSV into the form, matching column headers to known feature names.
- **Prediction** — the Analyze button calls `POST /predict/{model_name}` with the current feature values and renders the probability, confidence, threshold, and SHAP contribution breakdown.
- **Scenario testing** — the "Seeded Dataset Evaluation" panel calls `GET /scenario-tests/summary` and renders the TP/TN/FP/FN table and per-model confusion matrix grids.
- **Performance report** — the left detail panel shows ROC-AUC, PR-AUC, FP/FN counts, and the training confusion matrix for the selected model.

### Frontend — `frontend/itd/services/api.ts`

Axios-based API client with full TypeScript types for all request and response shapes. Key exports: `getModelsStatus`, `getModelMetrics`, `predictModel`, `getScenarioSummary`.

### ML — `ml/training_common.py`

Shared utilities used by all three training scripts:

- `load_dataset` — reads the CSV, parses each value to its correct type, and identifies constant columns to drop.
- `generate_oof_probabilities` — runs 5-fold stratified cross-validation to produce out-of-fold probability estimates used for threshold selection, avoiding test-set leakage.
- `select_threshold_for_recall` — sweeps candidate thresholds on OOF predictions and selects the highest-precision threshold that achieves ≥ the target recall (default 99%).
- `save_model_and_report` — persists the model bundle (model, vectorizer, threshold, metadata) to both `ml/models/` and `artifacts/models/`, and writes the full metrics JSON to `ml/reports/`.

### ML — Training scripts

All three training scripts follow the same pattern:

1. Load dataset via `training_common.load_dataset`.
2. Split 80% train / 20% test with `random_state=42` (no stratification on the split itself).
3. Load best hyperparameters from `reports/best_params.json` if present, otherwise use defaults.
4. Fit a `DictVectorizer` on the training features.
5. Train the classifier.
6. Generate OOF probabilities via 5-fold CV and select the decision threshold.
7. Evaluate on the held-out test set and save artefacts and metrics.

### ML — `ml/test_scenarios.py`

Command-line scenario tester. Loads the dataset, draws a stratified sample (25% malicious by default), runs each requested model over the sample, prints a console report with TP/TN/FP/FN and derived metrics (accuracy, precision, recall, F1), and optionally exports a consolidated `.md` or `.csv` report.

---

## 6. Description of Trained Model Files

### Artefact locations

| File | Purpose |
|---|---|
| `artifacts/models/xgboost_model.joblib` | Used by the backend in Docker and local deployment |
| `artifacts/models/lightgbm_model.joblib` | Used by the backend in Docker and local deployment |
| `artifacts/models/random_forest_model.joblib` | Used by the backend in Docker and local deployment |
| `ml/models/xgboost_model.joblib` | Training output, used by ML scripts |
| `ml/models/lightgbm_model.joblib` | Training output, used by ML scripts |
| `ml/models/random_forest_model.joblib` | Training output, used by ML scripts |

### Bundle structure

Each `.joblib` file is a Python dictionary with the following keys:

| Key | Type | Description |
|---|---|---|
| `model` | classifier object | Trained sklearn-compatible classifier |
| `vectorizer` | `DictVectorizer` | Fitted encoder — must be applied to every inference input |
| `target_column` | str | `"is_malicious"` |
| `constant_columns` | list[str] | Columns removed at training time (e.g. `late_exit_flag`) |
| `decision_threshold` | float | Tuned threshold for converting probability to binary label |
| `model_name` | str | `"xgboost"`, `"lightgbm"`, or `"random_forest"` |
| `numeric_columns` | list[str] | Feature names that must be parsed as numbers |

### Test-set performance (20% held-out, 23,723 records)

| Model | Threshold | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.0415 | 88.6% | 98.6% | 86.0% | 32.0% | 98.8% | 48.3% | 1262 | 19760 | 2686 | 15 |
| LightGBM | 0.0432 | 88.4% | 98.7% | 86.3% | 31.5% | 98.6% | 47.8% | 1259 | 19710 | 2736 | 18 |
| Random Forest | 0.1524 | 81.6% | 98.5% | 82.8% | 22.5% | 99.0% | 36.6% | 1264 | 18083 | 4363 | 13 |

The low precision and high recall are intentional. The decision threshold is tuned to minimise missed threats (false negatives) at the cost of more false alarms. At the default threshold of 0.5, XGBoost achieves 65.0% precision and 96.7% recall.

### Training configuration

| Setting | Value |
|---|---|
| Dataset size | 118,614 records |
| Train / test split | 80% / 20% (`random_state=42`) |
| Class balance (malicious) | 5.38% positive rate |
| Threshold selection method | Out-of-fold CV, maximise precision subject to recall ≥ 99% |
| Hyperparameter search | Optuna Bayesian optimisation, 50 trials, 5-fold CV, PR-AUC objective |
| Encoded feature count | 110 (after one-hot encoding and constant column removal) |

### Best hyperparameters

**XGBoost:**

| Parameter | Value |
|---|---|
| n_estimators | 580 |
| learning_rate | 0.0595 |
| max_depth | 8 |
| subsample | 0.7962 |
| colsample_bytree | 0.7280 |
| min_child_weight | 3 |
| gamma | 0.6843 |

**LightGBM:**

| Parameter | Value |
|---|---|
| n_estimators | 365 |
| learning_rate | 0.1037 |
| num_leaves | 39 |
| max_depth | 10 |
| subsample | 0.9642 |
| colsample_bytree | 0.5459 |
| min_child_samples | 49 |

**Random Forest:**

| Parameter | Value |
|---|---|
| n_estimators | 305 |
| max_depth | 22 |
| min_samples_split | 3 |
| min_samples_leaf | 2 |
| max_features | sqrt |

---

## 7. Screenshots Demonstrating Prototype Functionality

> Screenshots should be captured from a running instance and embedded here. Suggested captures:
>
> 1. **Landing page** — the ITD Portal login screen (`http://localhost:3000`).
> 2. **Dashboard — model selection** — the three model cards with accuracy, recall, precision, and F1 metrics.
> 3. **Dashboard — prediction result** — a completed threat assessment showing probability, confidence, and SHAP contribution breakdown for a malicious prediction.
> 4. **Dashboard — SHAP explanation** — the behavioral indicators panel showing the top risk drivers and their direction.
> 5. **Dashboard — scenario test results** — the seeded evaluation panel after clicking "RUN TEST", showing the TP/TN/FP/FN table and per-model confusion matrix grids.
> 6. **Dashboard — performance report** — the model details panel showing ROC-AUC, PR-AUC, and the training confusion matrix.

---

## 8. Troubleshooting Notes

**Backend returns `503 Model artifact not found`**
The model `.joblib` file is missing from `artifacts/models/`. Retrain the models using the training scripts in `ml/`, then confirm the files exist at `artifacts/models/xgboost_model.joblib` etc.

**Docker Compose fails with "read-only file system" on mount**
This occurs when two volume mounts are nested (e.g. mounting `artifacts/` and then `artifacts/data/` separately). The `docker-compose.yml` mounts `artifacts/models/` and `ml/data/` as sibling paths — ensure they are not nested under the same parent mount.

**Prediction fails with a 422 numeric validation error**
Numeric feature fields (`num_entries`, `total_printed_pages`, `employee_classification`, `total_files_burned`, etc.) must contain valid numbers. Text in a numeric field, or an empty string, will be rejected.

**Frontend cannot reach the backend**
Confirm the backend is running on port `8000`. If the frontend was started with a custom `NEXT_PUBLIC_API_BASE_URL`, ensure it matches the backend address. In Docker, inter-service communication uses the service name `backend` on port `8000`; do not set `NEXT_PUBLIC_API_BASE_URL` to `localhost` inside a container.

**SHAP explanations are missing or fail**
Confirm that `shap`, `xgboost`, and `lightgbm` are installed in the active Python environment. SHAP uses `TreeExplainer` for Random Forest and direct `pred_contribs` for XGBoost and LightGBM.

**scikit-learn version mismatch on model load**
The artefacts were serialised with scikit-learn 1.7.1. Loading them with a different version may produce a warning or error. Pin `scikit-learn==1.7.1` in your environment.

**Recall appears 100% in small scenario tests**
The decision thresholds are deliberately low (4–15%) to target 99% recall. On a small stratified sample (~12 malicious records out of 50), all malicious records typically score above even the 4% threshold, producing 0 false negatives. This is expected behaviour. Increase `--n` to see variance.
