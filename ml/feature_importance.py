import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.joblib"
TARGET_COLUMN = "is_malicious"
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42

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
            labels.append(int(float(row[TARGET_COLUMN])))
            feature_row = {}
            for column_name, raw_value in row.items():
                if column_name == TARGET_COLUMN:
                    continue
                parsed = parse_value(column_name, raw_value)
                feature_row[column_name] = parsed
                column_values.setdefault(column_name, set()).add(parsed)
            features.append(feature_row)

    constant_columns = {
        column_name for column_name, values in column_values.items() if len(values) <= 1
    }
    if constant_columns:
        for feature_row in features:
            for column_name in constant_columns:
                feature_row.pop(column_name, None)

    return features, labels, sorted(constant_columns)


def compute_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "decision_threshold": float(threshold),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def get_raw_feature_name(encoded_feature_name: str) -> str:
    if "=" in encoded_feature_name:
        return encoded_feature_name.split("=", 1)[0]
    return encoded_feature_name


def build_feature_groups(vectorizer) -> dict:
    feature_groups = {}
    encoded_feature_names = list(vectorizer.get_feature_names_out())
    for index, encoded_name in enumerate(encoded_feature_names):
        raw_name = get_raw_feature_name(encoded_name)
        group = feature_groups.setdefault(raw_name, {"indices": [], "encoded_columns": []})
        group["indices"].append(index)
        group["encoded_columns"].append(encoded_name)
    return feature_groups


def summarise_importance(values: list[float]) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    return float(values_array.mean()), float(values_array.std(ddof=0))


def compute_feature_importance(
    model,
    X_test,
    y_test,
    threshold: float,
    feature_groups: dict,
    repeats: int,
    seed: int,
    limit_features: int | None,
) -> tuple[dict, list[dict]]:
    baseline_probabilities = model.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test, baseline_probabilities, threshold)
    rng = np.random.default_rng(seed)

    feature_names = sorted(feature_groups)
    if limit_features is not None:
        feature_names = feature_names[:limit_features]

    importances = []
    for feature_name in feature_names:
        group = feature_groups[feature_name]
        false_negative_increases = []
        recall_drops = []
        pr_auc_drops = []
        precision_drops = []
        f1_drops = []

        for _ in range(repeats):
            permutation = rng.permutation(X_test.shape[0])
            X_permuted = X_test.copy()
            X_permuted[:, group["indices"]] = X_permuted[permutation][:, group["indices"]]
            permuted_probabilities = model.predict_proba(X_permuted)[:, 1]
            permuted_metrics = compute_metrics(y_test, permuted_probabilities, threshold)

            false_negative_increases.append(
                permuted_metrics["false_negatives"] - baseline_metrics["false_negatives"]
            )
            recall_drops.append(baseline_metrics["recall"] - permuted_metrics["recall"])
            pr_auc_drops.append(baseline_metrics["pr_auc"] - permuted_metrics["pr_auc"])
            precision_drops.append(baseline_metrics["precision"] - permuted_metrics["precision"])
            f1_drops.append(baseline_metrics["f1"] - permuted_metrics["f1"])

        fn_mean, fn_std = summarise_importance(false_negative_increases)
        recall_mean, recall_std = summarise_importance(recall_drops)
        pr_auc_mean, pr_auc_std = summarise_importance(pr_auc_drops)
        precision_mean, precision_std = summarise_importance(precision_drops)
        f1_mean, f1_std = summarise_importance(f1_drops)

        importances.append(
            {
                "feature": feature_name,
                "encoded_column_count": len(group["encoded_columns"]),
                "encoded_columns": group["encoded_columns"],
                "false_negative_increase_mean": fn_mean,
                "false_negative_increase_std": fn_std,
                "recall_drop_mean": recall_mean,
                "recall_drop_std": recall_std,
                "pr_auc_drop_mean": pr_auc_mean,
                "pr_auc_drop_std": pr_auc_std,
                "precision_drop_mean": precision_mean,
                "precision_drop_std": precision_std,
                "f1_drop_mean": f1_mean,
                "f1_drop_std": f1_std,
            }
        )

    importances.sort(
        key=lambda row: (
            row["false_negative_increase_mean"],
            row["recall_drop_mean"],
            row["pr_auc_drop_mean"],
            row["f1_drop_mean"],
        ),
        reverse=True,
    )
    return baseline_metrics, importances


def default_output_path(model_path: Path) -> Path:
    return BASE_DIR / "reports" / f"{model_path.stem}_feature_importance.json"


def main():
    parser = argparse.ArgumentParser(
        description="Estimate raw-feature importance by permuting encoded feature groups on the held-out test set."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        metavar="PATH",
        help="Path to a saved model artifact (default: ml/models/xgboost_model.joblib)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        metavar="PATH",
        help="Path to the dataset CSV used to train the model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to write the JSON report (default: ml/reports/<model>_feature_importance.json)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of random permutations per feature (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for permutations (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--limit-features",
        type=int,
        default=None,
        help="Only evaluate the first N raw features alphabetically; useful for quick smoke tests",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model}")
    if args.repeats <= 0:
        raise ValueError("--repeats must be at least 1.")
    if args.limit_features is not None and args.limit_features <= 0:
        raise ValueError("--limit-features must be at least 1 when provided.")

    features, labels, dropped_constant_columns = load_dataset(args.dataset)
    _, X_test_raw, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model_payload = joblib.load(args.model)
    model = model_payload["model"]
    vectorizer = model_payload["vectorizer"]
    decision_threshold = float(model_payload.get("decision_threshold", 0.5))

    X_test = np.asarray(vectorizer.transform(X_test_raw))
    feature_groups = build_feature_groups(vectorizer)
    baseline_metrics, importances = compute_feature_importance(
        model,
        X_test,
        y_test,
        decision_threshold,
        feature_groups,
        args.repeats,
        args.seed,
        args.limit_features,
    )

    output_path = args.output or default_output_path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_path": str(args.dataset.resolve()),
        "model_path": str(args.model.resolve()),
        "test_rows": len(X_test_raw),
        "decision_threshold": decision_threshold,
        "repeats": args.repeats,
        "seed": args.seed,
        "limit_features": args.limit_features,
        "dropped_constant_columns": dropped_constant_columns,
        "baseline_metrics": baseline_metrics,
        "raw_feature_count": len(feature_groups),
        "feature_importance": importances,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved feature importance report to: {output_path}")
    print("Top features by increased false negatives after permutation:")
    for row in importances[:10]:
        print(
            f"{row['feature']:<28} "
            f"FN {row['false_negative_increase_mean']:+.2f} "
            f"Recall {row['recall_drop_mean']:+.4f} "
            f"PR-AUC {row['pr_auc_drop_mean']:+.4f}"
        )


if __name__ == "__main__":
    main()
