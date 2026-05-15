# COS720-Project

Repository structure:

- `frontend/` for the UI application
- `backend/` for APIs and server-side logic
- `ml/` for datasets, training code, and model reports
- `artifacts/` for trained model artifacts shared with the backend

## ML Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ml\requirements.txt
python ml\train_random_forest.py
```

Training writes model bundles to both `ml/models/` and `artifacts/models/`.
The backend reads from `artifacts/models/` when run through Docker Compose,
preferring XGBoost, then LightGBM, then Random Forest. Set `MODEL_PATH` to use
a specific artifact.
