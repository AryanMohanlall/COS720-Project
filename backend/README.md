# Backend

FastAPI service for serving predictions from trained ML artifacts.

By default the backend reads from `artifacts/models/`, preferring XGBoost, then
LightGBM, then Random Forest. Override this with `MODEL_PATH` when needed.

Useful endpoints:

- `GET /model/status` checks the configured artifact path.
- `POST /predict` accepts `{"features": {...}}` using the raw training column names.
