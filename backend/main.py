import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
shap_explainers_by_model_id: dict[int, Any] = {}


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


def raw_feature_name(encoded_feature_name: str) -> str:
    return encoded_feature_name.split("=", 1)[0]


def positive_class_shap_values(shap_values: Any) -> np.ndarray:
    values = np.asarray(shap_values)

    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values[0])

    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]

    if values.ndim == 2:
        return values[0]

    return values


def positive_class_base_value(expected_value: Any) -> float | None:
    if expected_value is None:
        return None

    values = np.asarray(expected_value)
    if values.ndim == 0:
        return float(values)
    if values.size > 1:
        return float(values.reshape(-1)[1])
    return float(values.reshape(-1)[0])


def get_shap_explainer(model: Any):
    model_id = id(model)
    if model_id not in shap_explainers_by_model_id:
        try:
            import shap
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="SHAP is not installed. Install backend requirements before requesting explanations.",
            ) from exc

        shap_explainers_by_model_id[model_id] = shap.TreeExplainer(model)

    return shap_explainers_by_model_id[model_id]


def xgboost_contributions(model: Any, vectorized, encoded_feature_names: list[str]):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="XGBoost is not installed. Install backend requirements before requesting explanations.",
        ) from exc

    booster = model.get_booster()
    matrix = xgb.DMatrix(vectorized, feature_names=encoded_feature_names)
    contributions = booster.predict(matrix, pred_contribs=True)[0]

    return contributions[:-1], float(contributions[-1]), "xgboost_pred_contribs"


def lightgbm_contributions(model: Any, vectorized):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        contributions = np.asarray(model.predict(vectorized, pred_contrib=True))[0]
    return contributions[:-1], float(contributions[-1]), "lightgbm_pred_contrib"


def shap_tree_contributions(model: Any, vectorized):
    explainer = get_shap_explainer(model)
    shap_values = positive_class_shap_values(explainer.shap_values(vectorized))
    base_value = positive_class_base_value(getattr(explainer, "expected_value", None))

    return shap_values, base_value, "shap_tree_explainer"


def model_contributions(
    bundle: dict[str, Any],
    vectorized,
    encoded_feature_names: list[str],
):
    model = bundle["model"]
    model_name = bundle.get("model_name")

    if model_name == "xgboost":
        return xgboost_contributions(model, vectorized, encoded_feature_names)
    if model_name == "lightgbm":
        return lightgbm_contributions(model, vectorized)

    return shap_tree_contributions(model, vectorized)


def explain_prediction(bundle: dict[str, Any], vectorized, features: dict[str, Any]):
    vectorizer = bundle["vectorizer"]
    encoded_feature_names = list(vectorizer.get_feature_names_out())
    vector_values = np.asarray(vectorized).reshape(-1)
    shap_values, base_value, method = model_contributions(
        bundle,
        vectorized,
        encoded_feature_names,
    )

    encoded_contributions = []
    raw_contributions: dict[str, dict[str, Any]] = {}

    for encoded_name, encoded_value, shap_value in zip(
        encoded_feature_names,
        vector_values,
        shap_values,
        strict=False,
    ):
        raw_name = raw_feature_name(encoded_name)
        contribution = float(shap_value)
        encoded_contributions.append(
            {
                "feature": encoded_name,
                "raw_feature": raw_name,
                "value": float(encoded_value),
                "shap_value": contribution,
                "abs_shap_value": abs(contribution),
                "direction": "increases_risk" if contribution >= 0 else "decreases_risk",
            }
        )

        current = raw_contributions.setdefault(
            raw_name,
            {
                "feature": raw_name,
                "value": features.get(raw_name),
                "shap_value": 0.0,
                "abs_shap_value": 0.0,
            },
        )
        current["shap_value"] += contribution

    for contribution in raw_contributions.values():
        contribution["abs_shap_value"] = abs(contribution["shap_value"])
        contribution["direction"] = (
            "increases_risk" if contribution["shap_value"] >= 0 else "decreases_risk"
        )

    sorted_raw_contributions = sorted(
        raw_contributions.values(),
        key=lambda contribution: contribution["abs_shap_value"],
        reverse=True,
    )
    sorted_encoded_contributions = sorted(
        encoded_contributions,
        key=lambda contribution: contribution["abs_shap_value"],
        reverse=True,
    )

    return {
        "method": method,
        "output": "raw_model_score",
        "base_value": base_value,
        "top_contributions": sorted_raw_contributions[:10],
        "raw_feature_contributions": sorted_raw_contributions,
        "encoded_feature_contributions": sorted_encoded_contributions[:20],
    }


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
    explanation = explain_prediction(bundle, vectorized, features)

    return {
        "model_name": bundle.get("model_name"),
        "model_path": str(path) if path is not None else None,
        "probability": probability,
        "decision_threshold": threshold,
        "prediction": int(probability >= threshold),
        "explanation": explanation,
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
