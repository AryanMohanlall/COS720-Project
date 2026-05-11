import csv
import json
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
RF_METRICS_PATH = BASE_DIR / "reports" / "random_forest_metrics.json"
TARGET_COLUMN = "is_malicious"

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
        col for col, vals in column_values.items() if len(vals) <= 1
    }
    if constant_columns:
        for feature_row in features:
            for col in constant_columns:
                feature_row.pop(col, None)

    return features, labels, sorted(constant_columns)


def compute_metrics(y_test, predictions, probabilities):
    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def save_model_and_report(model, vectorizer, constant_columns, metrics_extra, model_path, report_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "target_column": TARGET_COLUMN,
            "constant_columns": constant_columns,
        },
        model_path,
    )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_extra, handle, indent=2)


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    features, labels, constant_columns = load_dataset(DATASET_PATH)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    neg = sum(1 for y in y_train if y == 0)
    pos = sum(1 for y in y_train if y == 1)
    scale_pos_weight = neg / pos  # ~17.6 for this dataset

    shared_meta = {
        "dataset_path": str(DATASET_PATH),
        "train_rows": len(X_train_raw),
        "test_rows": len(X_test_raw),
        "feature_count_after_encoding": int(X_train.shape[1]),
        "dropped_constant_columns": constant_columns,
        "positive_rate_train": pos / (pos + neg),
        "positive_rate_test": sum(y_test) / len(y_test),
    }

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr",
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = compute_metrics(y_test, xgb_preds, xgb_probs)

    save_model_and_report(
        xgb_model, vectorizer, constant_columns,
        {**shared_meta, **xgb_metrics},
        BASE_DIR / "models" / "xgboost_model.joblib",
        BASE_DIR / "reports" / "xgboost_metrics.json",
    )

    print("\nXGBoost classification report:")
    print(classification_report(
        y_test, xgb_preds, digits=4, zero_division=0,
        target_names=["not_malicious", "malicious"],
    ))

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    print("Training LightGBM...")
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_preds = lgbm_model.predict(X_test)
    lgbm_metrics = compute_metrics(y_test, lgbm_preds, lgbm_probs)

    save_model_and_report(
        lgbm_model, vectorizer, constant_columns,
        {**shared_meta, **lgbm_metrics},
        BASE_DIR / "models" / "lightgbm_model.joblib",
        BASE_DIR / "reports" / "lightgbm_metrics.json",
    )

    print("\nLightGBM classification report:")
    print(classification_report(
        y_test, lgbm_preds, digits=4, zero_division=0,
        target_names=["not_malicious", "malicious"],
    ))

    # ------------------------------------------------------------------
    # Comparison table (includes RF if its metrics file exists)
    # ------------------------------------------------------------------
    rf_metrics = {}
    if RF_METRICS_PATH.exists():
        with RF_METRICS_PATH.open(encoding="utf-8") as f:
            rf_metrics = json.load(f)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    has_rf = bool(rf_metrics)
    if has_rf:
        print(f"{'Metric':<20} {'Random Forest':>14} {'XGBoost':>10} {'LightGBM':>10}")
        print("-" * 57)
    else:
        print(f"{'Metric':<20} {'XGBoost':>10} {'LightGBM':>10}")
        print("-" * 42)

    for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        xgb_val = xgb_metrics[metric]
        lgbm_val = lgbm_metrics[metric]
        if has_rf:
            rf_val = rf_metrics.get(metric, float("nan"))
            print(f"{metric:<20} {rf_val:>14.4f} {xgb_val:>10.4f} {lgbm_val:>10.4f}")
        else:
            print(f"{metric:<20} {xgb_val:>10.4f} {lgbm_val:>10.4f}")

    print("=" * 65)
    print(f"\nXGBoost model  -> models/xgboost_model.joblib")
    print(f"LightGBM model -> models/lightgbm_model.joblib")


if __name__ == "__main__":
    main()
