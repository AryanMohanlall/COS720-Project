import csv
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
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


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
REPORT_OUTPUT_PATH = BASE_DIR / "reports" / "random_forest_metrics.json"
TARGET_COLUMN = "is_malicious"

# Numeric columns should remain numeric instead of being one-hot encoded as text.
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

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    metrics = {
        "dataset_path": str(DATASET_PATH),
        "train_rows": len(X_train_raw),
        "test_rows": len(X_test_raw),
        "feature_count_after_encoding": int(X_train.shape[1]),
        "dropped_constant_columns": constant_columns,
        "positive_rate_train": sum(y_train) / len(y_train),
        "positive_rate_test": sum(y_test) / len(y_test),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "target_column": TARGET_COLUMN,
            "constant_columns": constant_columns,
        },
        MODEL_OUTPUT_PATH,
    )

    with REPORT_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Random Forest training complete.")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Encoded feature count: {metrics['feature_count_after_encoding']}")
    print(f"Dropped constant columns: {constant_columns or 'None'}")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
            target_names=["not_malicious", "malicious"],
        )
    )
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
