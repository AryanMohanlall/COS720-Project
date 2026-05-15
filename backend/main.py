import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    APP_DIR / "artifacts" / "models" / "xgboost_model.joblib",
    APP_DIR / "artifacts" / "models" / "lightgbm_model.joblib",
    APP_DIR / "artifacts" / "models" / "random_forest_model.joblib",
    APP_DIR.parent / "artifacts" / "models" / "xgboost_model.joblib",
    APP_DIR.parent / "artifacts" / "models" / "lightgbm_model.joblib",
    APP_DIR.parent / "artifacts" / "models" / "random_forest_model.joblib",
    APP_DIR.parent / "ml" / "models" / "xgboost_model.joblib",
    APP_DIR.parent / "ml" / "models" / "lightgbm_model.joblib",
    APP_DIR.parent / "ml" / "models" / "random_forest_model.joblib",
]
MODEL_PATH = Path(os.getenv("MODEL_PATH", "")).expanduser() if os.getenv("MODEL_PATH") else None

model_bundle: dict[str, Any] | None = None
loaded_model_path: Path | None = None


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw feature values keyed by training column name.")


def parse_feature_value(column_name: str, value: Any, numeric_columns: set[str]) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        value = cleaned

    if column_name in numeric_columns:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{column_name}' must be numeric.",
            ) from exc
        return int(numeric_value) if numeric_value.is_integer() else numeric_value

    return value


def resolve_model_path() -> Path:
    if MODEL_PATH:
        return MODEL_PATH

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_MODEL_CANDIDATES[0]


def load_model_bundle() -> dict[str, Any]:
    global model_bundle, loaded_model_path

    if model_bundle is not None:
        return model_bundle

    path = resolve_model_path()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artifact not found at {path}. Train a model first or set MODEL_PATH "
                "to a specific .joblib artifact."
            ),
        )

    model_bundle = joblib.load(path)
    loaded_model_path = path
    return model_bundle


@app.get("/")
async def root():
    return {"message": "Backend API is running"}


@app.get("/model/status")
async def model_status():
    path = resolve_model_path()
    loaded = model_bundle is not None
    return {
        "configured_model_path": str(path),
        "exists": path.exists(),
        "loaded": loaded,
        "model_name": model_bundle.get("model_name") if loaded else None,
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    bundle = load_model_bundle()
    numeric_columns = set(bundle.get("numeric_columns", []))
    features = {
        column: parse_feature_value(column, value, numeric_columns)
        for column, value in request.features.items()
    }

    for column in bundle.get("constant_columns", []):
        features.pop(column, None)

    vectorized = bundle["vectorizer"].transform([features])
    probability = float(bundle["model"].predict_proba(vectorized)[0, 1])
    threshold = float(bundle.get("decision_threshold", 0.5))

    return {
        "model_name": bundle.get("model_name"),
        "model_path": str(loaded_model_path),
        "probability": probability,
        "decision_threshold": threshold,
        "prediction": int(probability >= threshold),
    }
