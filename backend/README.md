# Backend

FastAPI service for serving predictions from trained ML artifacts.

By default the backend reads from `artifacts/models/`, preferring XGBoost, then
LightGBM, then Random Forest. Override this with `MODEL_PATH` when needed.

Useful endpoints:

- `GET /model/status` checks the configured default artifact path.
- `GET /models/status` checks all supported model artifacts and metric reports.
- `GET /metrics/{model_name}` returns the saved metrics report for a model.
- `POST /predict` predicts with the default configured model.
- `POST /predict/{model_name}` predicts with a specific model.

Supported `model_name` values:

- `xgboost`
- `lightgbm`
- `random_forest`

Prediction endpoints accept `{"features": {...}}` using the raw training column names.
Prediction responses include a SHAP explanation with the top raw feature
contributions for the selected model.
