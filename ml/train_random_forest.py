import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
REPORT_OUTPUT_PATH = BASE_DIR / "reports" / "random_forest_metrics.json"
BEST_PARAMS_PATH = BASE_DIR / "reports" / "best_params.json"
TARGET_COLUMN = "is_malicious"

N_TRIALS = 50
CV_FOLDS = 5
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


def load_best_params(params_path: Path) -> dict | None:
    if not params_path.exists():
        return None
    with params_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("random_forest", {}).get("best_params")


def tune_random_forest(X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": 1,
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params


def compute_metrics(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "decision_threshold": None,
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


def generate_oof_probabilities(feature_rows, labels, model_builder) -> np.ndarray:
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
        probabilities[valid_idx] = model.predict_proba(X_fold_valid)[:, 1]

    return probabilities


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest classifier")
    parser.add_argument(
        "--params",
        type=Path,
        default=BEST_PARAMS_PATH,
        metavar="PATH",
        help="Path to best_params.json produced by tune_hyperparameters.py (default: reports/best_params.json)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help=(
            "Minimum recall to target when selecting the decision threshold from "
            f"out-of-fold training predictions (default: {DEFAULT_TARGET_RECALL})"
        ),
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not 0 < args.target_recall <= 1:
        raise ValueError("--target-recall must be between 0 and 1.")

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

    best_params = load_best_params(args.params)
    if best_params is not None:
        print(f"\nLoaded pre-tuned Random Forest params from {args.params}")
        print(f"Params: {best_params}")
    else:
        print(f"\nTuning Random Forest ({N_TRIALS} Optuna trials, {CV_FOLDS}-fold CV)...")
        best_params = tune_random_forest(X_train, y_train)
        print(f"Best Random Forest params: {best_params}")

    print(f"Selecting Random Forest decision threshold for target recall {args.target_recall:.2%}...")
    oof_probabilities = generate_oof_probabilities(
        X_train_raw,
        y_train,
        lambda _: RandomForestClassifier(
            **best_params,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    )
    selected_threshold, threshold_reason = select_threshold_for_recall(
        y_train,
        oof_probabilities,
        args.target_recall,
    )
    print(
        "Selected Random Forest threshold "
        f"{selected_threshold['decision_threshold']:.4f} "
        f"(OOF recall={selected_threshold['recall']:.4f}, "
        f"precision={selected_threshold['precision']:.4f}, "
        f"false_negatives={selected_threshold['false_negatives']})"
    )

    print("Training Random Forest with best params...")
    model = RandomForestClassifier(
        **best_params,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = compute_probability_metrics(
        y_test,
        probabilities,
        selected_threshold["decision_threshold"],
    )
    predictions = (probabilities >= metrics["decision_threshold"]).astype(int)
    default_threshold_metrics = compute_probability_metrics(y_test, probabilities, 0.5)

    report = {
        "dataset_path": str(DATASET_PATH),
        "train_rows": len(X_train_raw),
        "test_rows": len(X_test_raw),
        "feature_count_after_encoding": int(X_train.shape[1]),
        "dropped_constant_columns": constant_columns,
        "positive_rate_train": sum(y_train) / len(y_train),
        "positive_rate_test": sum(y_test) / len(y_test),
        "best_params": best_params,
        **metrics,
        "default_threshold_metrics": default_threshold_metrics,
        "threshold_selection": {
            "selection_basis": "out_of_fold_training_predictions",
            "target_recall": args.target_recall,
            "selection_rule": threshold_reason,
            "selected_training_metrics": selected_threshold,
            "oof_threshold_sweep": build_threshold_sweep(
                y_train,
                oof_probabilities,
                build_report_thresholds(selected_threshold["decision_threshold"]),
            ),
            "test_threshold_sweep": build_threshold_sweep(
                y_test,
                probabilities,
                build_report_thresholds(selected_threshold["decision_threshold"]),
            ),
        },
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "target_column": TARGET_COLUMN,
            "constant_columns": constant_columns,
            "decision_threshold": metrics["decision_threshold"],
        },
        MODEL_OUTPUT_PATH,
    )

    with REPORT_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nRandom Forest training complete.")
    print(f"Train rows: {report['train_rows']}")
    print(f"Test rows: {report['test_rows']}")
    print(f"Encoded feature count: {report['feature_count_after_encoding']}")
    print(f"Dropped constant columns: {constant_columns or 'None'}")
    print(classification_report(
        y_test, predictions, digits=4, zero_division=0,
        target_names=["not_malicious", "malicious"],
    ))
    print(f"Decision threshold: {metrics['decision_threshold']:.4f}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
