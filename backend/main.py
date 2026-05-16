import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "xgboost": "xgboost_model.joblib",
    "lightgbm": "lightgbm_model.joblib",
    "random_forest": "random_forest_model.joblib",
}
METRIC_FILES = {
    "xgboost": "xgboost_metrics.json",
    "lightgbm": "lightgbm_metrics.json",
    "random_forest": "random_forest_metrics.json",
}
MODEL_ALIASES = {
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "rf": "random_forest",
    "random-forest": "random_forest",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
}
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
model_bundles_by_name: dict[str, dict[str, Any]] = {}
loaded_model_paths_by_name: dict[str, Path] = {}


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw feature values keyed by training column name.")


def normalize_model_name(model_name: str) -> str:
    normalized = MODEL_ALIASES.get(model_name.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(MODEL_FILES))
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Supported models: {supported}.",
        )
    return normalized


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


def model_candidates(model_name: str) -> list[Path]:
    filename = MODEL_FILES[model_name]
    return [
        APP_DIR / "artifacts" / "models" / filename,
        APP_DIR.parent / "artifacts" / "models" / filename,
        APP_DIR.parent / "ml" / "models" / filename,
    ]


def resolve_named_model_path(model_name: str) -> Path:
    for candidate in model_candidates(model_name):
        if candidate.exists():
            return candidate

    return model_candidates(model_name)[0]


def resolve_metrics_path(model_name: str) -> Path:
    filename = METRIC_FILES[model_name]
    candidates = [
        APP_DIR / "reports" / filename,
        APP_DIR.parent / "ml" / "reports" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


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


def load_named_model_bundle(model_name: str) -> tuple[dict[str, Any], Path]:
    normalized = normalize_model_name(model_name)

    if normalized in model_bundles_by_name:
        return model_bundles_by_name[normalized], loaded_model_paths_by_name[normalized]

    path = resolve_named_model_path(normalized)
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact for '{normalized}' not found at {path}. Train the model first.",
        )

    bundle = joblib.load(path)
    model_bundles_by_name[normalized] = bundle
    loaded_model_paths_by_name[normalized] = path
    return bundle, path


def run_prediction(bundle: dict[str, Any], path: Path | None, request: PredictionRequest):
    numeric_columns = set(bundle.get("numeric_columns", []))
    features = {
        column: parse_feature_value(column, value, numeric_columns)
        for column, value in request.features.items()
    }

    for column in bundle.get("constant_columns", []):
        features.pop(column, None)

    vectorized = bundle["vectorizer"].transform([features])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        probability = float(bundle["model"].predict_proba(vectorized)[0, 1])
    threshold = float(bundle.get("decision_threshold", 0.5))

    return {
        "model_name": bundle.get("model_name"),
        "model_path": str(path) if path is not None else None,
        "probability": probability,
        "decision_threshold": threshold,
        "prediction": int(probability >= threshold),
    }


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


@app.get("/models/status")
async def models_status():
    return {
        model_name: {
            "model_path": str(resolve_named_model_path(model_name)),
            "model_exists": resolve_named_model_path(model_name).exists(),
            "metrics_path": str(resolve_metrics_path(model_name)),
            "metrics_exists": resolve_metrics_path(model_name).exists(),
            "loaded": model_name in model_bundles_by_name,
        }
        for model_name in MODEL_FILES
    }


@app.get("/metrics/{model_name}")
async def model_metrics(model_name: str):
    normalized = normalize_model_name(model_name)
    path = resolve_metrics_path(normalized)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics report for '{normalized}' not found at {path}. Train the model first.",
        )

    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    return {
        "model_name": normalized,
        "metrics_path": str(path),
        "metrics": metrics,
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    bundle = load_model_bundle()
    return run_prediction(bundle, loaded_model_path, request)


@app.post("/predict/{model_name}")
async def predict_with_named_model(model_name: str, request: PredictionRequest):
    bundle, path = load_named_model_bundle(model_name)
    return run_prediction(bundle, path, request)
