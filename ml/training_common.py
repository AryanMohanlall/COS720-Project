import csv
import json
import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
RF_METRICS_PATH = BASE_DIR / "reports" / "random_forest_metrics.json"
BEST_PARAMS_PATH = BASE_DIR / "reports" / "best_params.json"
TARGET_COLUMN = "is_malicious"

CV_FOLDS = 5
N_TRIALS = 50
DEFAULT_TARGET_RECALL = 0.99

NUMERIC_COLUMNS = {
    "employee_seniority_years",
    "is_contractor",
    "employee_classification",
    "has_foreign_citizenship",
    "has_criminal_record",
    "has_medical_history",
    "total_printed_pages",
    "num_printed_pages_off_hours",
    "total_files_burned",
    "burned_from_other",
    "is_abroad",
    "trip_day_number",
    "hostility_country_level",
    "num_entries",
    "num_unique_campus",
    "late_exit_flag",
    "entry_during_weekend",
}


def parse_value(column_name: str, value: str):
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "":
        return None
    if column_name in NUMERIC_COLUMNS:
        numeric_value = float(cleaned)
        return int(numeric_value) if numeric_value.is_integer() else numeric_value
    return cleaned


def load_dataset(dataset_path: Path):
    features = []
    labels = []
    column_values = {}

    with dataset_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(float(row[TARGET_COLUMN]))
            feature_row = {}
            for column_name, raw_value in row.items():
                if column_name == TARGET_COLUMN:
                    continue
                parsed = parse_value(column_name, raw_value)
                feature_row[column_name] = parsed
                column_values.setdefault(column_name, set()).add(parsed)
            features.append(feature_row)
            labels.append(label)

    constant_columns = {
        column_name for column_name, values in column_values.items() if len(values) <= 1
    }
    if constant_columns:
        for feature_row in features:
            for column_name in constant_columns:
                feature_row.pop(column_name, None)

    return features, labels, sorted(constant_columns)


def load_best_params(params_path: Path, model_key: str) -> dict | None:
    if not params_path.exists():
        return None
    with params_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get(model_key, {}).get("best_params")


def compute_metrics(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "decision_threshold": None,
        "accuracy": accuracy_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "predicted_positive_rate": float(np.mean(predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def compute_probability_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    metrics = compute_metrics(y_true, predictions, probabilities)
    metrics["decision_threshold"] = float(threshold)
    return metrics


def build_threshold_sweep(y_true, probabilities, thresholds: list[float]) -> list[dict]:
    return [compute_probability_metrics(y_true, probabilities, threshold) for threshold in thresholds]


def generate_candidate_thresholds(probabilities) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, 201)
    thresholds = {0.5}
    for quantile in quantiles:
        thresholds.add(float(np.quantile(probabilities, quantile)))
    thresholds.update(float(value) for value in np.linspace(0.05, 0.95, 19))
    return sorted(min(1.0, max(0.0, threshold)) for threshold in thresholds)


def select_threshold_for_recall(y_true, probabilities, target_recall: float) -> tuple[dict, str]:
    threshold_metrics = build_threshold_sweep(
        y_true,
        probabilities,
        generate_candidate_thresholds(probabilities),
    )
    candidates = [row for row in threshold_metrics if row["recall"] >= target_recall]
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (row["precision"], row["f1"], row["decision_threshold"]),
        )
        reason = (
            f"highest precision among thresholds with recall >= {target_recall:.2%}, "
            "breaking ties with F1 and then a higher threshold"
        )
    else:
        selected = max(
            threshold_metrics,
            key=lambda row: (row["recall"], row["precision"], row["f1"], row["decision_threshold"]),
        )
        reason = (
            f"no threshold reached recall >= {target_recall:.2%}; "
            "selected the highest-recall threshold, breaking ties with precision, F1, and then a higher threshold"
        )
    return selected, reason


def build_report_thresholds(selected_threshold: float) -> list[float]:
    thresholds = {selected_threshold}
    thresholds.update(float(value) for value in np.linspace(0.1, 0.9, 9))
    return sorted(min(1.0, max(0.0, threshold)) for threshold in thresholds)


def predict_positive_probabilities(model, X) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        return model.predict_proba(X)[:, 1]


def generate_oof_probabilities(feature_rows, labels, model_builder: Callable) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    labels_array = np.asarray(labels)
    probabilities = np.zeros(len(labels_array), dtype=float)
    rows = list(feature_rows)

    for train_idx, valid_idx in cv.split(np.zeros(len(labels_array)), labels_array):
        vectorizer = DictVectorizer(sparse=False)
        train_rows = [rows[index] for index in train_idx]
        valid_rows = [rows[index] for index in valid_idx]
        X_fold_train = vectorizer.fit_transform(train_rows)
        X_fold_valid = vectorizer.transform(valid_rows)
        y_fold_train = labels_array[train_idx]

        model = model_builder(y_fold_train)
        model.fit(X_fold_train, y_fold_train)
        probabilities[valid_idx] = predict_positive_probabilities(model, X_fold_valid)

    return probabilities


def compute_scale_pos_weight(labels) -> float:
    labels_array = np.asarray(labels)
    positives = int(labels_array.sum())
    negatives = int(len(labels_array) - positives)
    if positives == 0:
        return 1.0
    return negatives / positives


def build_shared_meta(dataset_path, X_train_raw, X_test_raw, y_train, y_test, X_train, constant_columns):
    positives = int(np.sum(y_train))
    negatives = int(len(y_train) - positives)
    return {
        "dataset_path": str(dataset_path),
        "train_rows": len(X_train_raw),
        "test_rows": len(X_test_raw),
        "feature_count_after_encoding": int(X_train.shape[1]),
        "dropped_constant_columns": constant_columns,
        "positive_rate_train": positives / (positives + negatives),
        "positive_rate_test": sum(y_test) / len(y_test),
    }


def save_model_and_report(
    model,
    vectorizer,
    constant_columns,
    decision_threshold,
    metrics_extra,
    model_path,
    report_path,
    model_name,
):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = REPO_ROOT / "artifacts" / "models" / model_path.name
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "target_column": TARGET_COLUMN,
        "constant_columns": constant_columns,
        "decision_threshold": float(decision_threshold),
        "model_name": model_name,
        "numeric_columns": sorted(NUMERIC_COLUMNS),
    }
    joblib.dump(artifact, model_path)
    joblib.dump(artifact, artifact_path)

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_extra, handle, indent=2)

    return artifact_path


def format_metric_value(metric: str, value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if metric in {"false_negatives", "false_positives"}:
        return str(int(value))
    return f"{float(value):.4f}"
