# Technical Documentation

## 1. System Overview

The Insider Threat Detection prototype is a web-based system that classifies employee behaviour as either malicious or not malicious. The system consists of a Next.js frontend dashboard, a FastAPI backend, and an offline machine-learning training pipeline. Users can enter behavioural indicators manually or load them from a CSV file. The selected trained model then produces a malicious probability, applies a saved decision threshold, and returns a classification with SHAP-based risk drivers.

The main supported models are XGBoost, LightGBM, and Random Forest. Each model is trained on the cleaned insider-threat dataset and saved as a `.joblib` artefact containing the trained classifier, vectorizer, decision threshold, constant columns, and numeric feature list.

## 2. Software Requirements and Dependencies

Required software:

- Python 3.10 or later
- Node.js 18 or later
- npm
- Docker and Docker Compose, optional for containerised execution

Backend dependencies are listed in `backend/requirements.txt`:

- `fastapi`
- `uvicorn[standard]`
- `joblib`
- `scikit-learn==1.7.1`
- `xgboost`
- `lightgbm`
- `shap`

ML dependencies are listed in `ml/requirements.txt`:

- `scikit-learn==1.7.1`
- `xgboost`
- `lightgbm`
- `shap`
- `matplotlib`
- `optuna`

Frontend dependencies are listed in `frontend/itd/package.json`:

- Next.js
- React
- React DOM
- Axios
- Tailwind CSS
- TypeScript
- ESLint

## 3. Installation Instructions

Create and activate a Python virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
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

The trained model files should exist in either `artifacts/models/` or `ml/models/`. If they are missing, retrain the models using the ML scripts.

## 4. Deployment and Execution Instructions

To run the backend locally from the repository root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

To run the frontend locally:

```bash
cd frontend/itd
npm run dev
```

Open the dashboard at:

```text
http://localhost:3000
```

The backend API runs at:

```text
http://localhost:8000
```

Useful backend endpoints:

- `GET /model/status`
- `GET /models/status`
- `GET /metrics/{model_name}`
- `POST /predict`
- `POST /predict/{model_name}`

Supported model names are:

- `xgboost`
- `lightgbm`
- `random_forest`

To run the system with Docker Compose:

```bash
docker compose up --build
```

The Docker setup starts the backend on port `8000` and the frontend on port `3000`. The backend reads model artefacts from the mounted `artifacts/` directory.

To retrain the models:

```bash
cd ml
python tune_hyperparameters.py
python train_xgboost.py --target-recall 0.99
python train_lightgbm.py --target-recall 0.99
python train_random_forest.py --target-recall 0.99
```

To regenerate XGBoost SHAP reports and feature importance:

```bash
python shap_analysis.py --model xgboost
python feature_importance.py --model models/xgboost_model.joblib --repeats 5
```

## 5. Source Code Documentation

The repository is organised as follows:

- `frontend/itd/`: Next.js dashboard used by the analyst.
- `backend/`: FastAPI prediction and metrics API.
- `ml/`: training scripts, SHAP analysis, feature importance, reports, and local model files.
- `artifacts/models/`: model artefacts used by the backend deployment.
- `ml/reports/`: saved metrics, threshold reports, feature-importance reports, and SHAP outputs.

Important frontend files:

- `frontend/itd/app/page.tsx`: landing page and dashboard entry point.
- `frontend/itd/app/(main)/dashboard/page.tsx`: main prototype dashboard. It handles model selection, manual feature input, CSV parsing, prediction requests, metric display, and SHAP explanation rendering.
- `frontend/itd/services/api.ts`: Axios API client and TypeScript response types for backend communication.

Important backend files:

- `backend/main.py`: FastAPI application. It loads model artefacts, validates feature values, applies the saved vectorizer, generates model probabilities, applies decision thresholds, computes SHAP contributions, and returns prediction responses.
- `backend/requirements.txt`: backend Python dependencies.

Important ML files:

- `ml/training_common.py`: shared dataset loading, preprocessing, metric calculation, threshold selection, and model-report saving utilities.
- `ml/tune_hyperparameters.py`: Optuna hyperparameter tuning for XGBoost, LightGBM, and Random Forest using PR-AUC.
- `ml/train_xgboost.py`: trains and saves the XGBoost classifier.
- `ml/train_lightgbm.py`: trains and saves the LightGBM classifier.
- `ml/train_random_forest.py`: trains and saves the Random Forest classifier.
- `ml/shap_analysis.py`: computes global SHAP explanations and aggregates encoded feature contributions back to raw feature names.
- `ml/feature_importance.py`: computes permutation-based feature importance on the held-out test set.

## 6. Description of Trained Model Files

The trained model artefacts are saved as `.joblib` files. Each artefact is a self-contained Python object used by the backend for prediction.

Model artefacts:

- `ml/models/xgboost_model.joblib`
- `ml/models/lightgbm_model.joblib`
- `ml/models/random_forest_model.joblib`
- `artifacts/models/xgboost_model.joblib`
- `artifacts/models/lightgbm_model.joblib`
- `artifacts/models/random_forest_model.joblib`

Each model bundle contains:

- `model`: trained classifier object.
- `vectorizer`: fitted `DictVectorizer` used for one-hot encoding.
- `target_column`: target field, `is_malicious`.
- `constant_columns`: zero-variance columns removed during training.
- `decision_threshold`: tuned classification threshold.
- `model_name`: model identifier.
- `numeric_columns`: list of numeric feature names.

The backend uses these artefacts to reproduce the same preprocessing used during training. Raw input features are parsed, constant columns are removed, the saved vectorizer transforms the input, the selected model predicts a malicious probability, and the saved threshold converts the probability into a binary label.

The Random Forest artefact is large and should not be committed directly to GitHub because it exceeds GitHub's normal file-size limit. Model files should be stored locally, shared through deployment artefacts, or managed through Git LFS if remote versioning is required.

## 8. Troubleshooting Notes

If the backend returns `503 Model artifact not found`, confirm that the selected model file exists in `artifacts/models/` or `ml/models/`. Retrain the models if the artefacts are missing.

If prediction fails with a numeric validation error, check that numeric fields such as `num_entries`, `total_printed_pages`, `employee_classification`, and `total_files_burned` contain valid numbers.

If the frontend cannot reach the backend, confirm that the backend is running on port `8000` and that the frontend API base URL points to `http://localhost:8000`.

If SHAP explanations fail, confirm that `shap`, `xgboost`, and `lightgbm` are installed in the active Python environment.

If dependency issues occur, recreate the virtual environment and reinstall dependencies from the requirements files.
