import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
RF_METRICS_PATH = BASE_DIR / "reports" / "random_forest_metrics.json"
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
        col for col, vals in column_values.items() if len(vals) <= 1
    }
    if constant_columns:
        for feature_row in features:
            for col in constant_columns:
                feature_row.pop(col, None)

    return features, labels, sorted(constant_columns)


def load_best_params(params_path: Path, model_key: str) -> dict | None:
    if not params_path.exists():
        return None
    with params_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get(model_key, {}).get("best_params")


def compute_metrics(y_test, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "decision_threshold": None,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "predicted_positive_rate": float(np.mean(predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def compute_scale_pos_weight(labels) -> float:
    labels_array = np.asarray(labels)
    positives = int(labels_array.sum())
    negatives = int(len(labels_array) - positives)
    if positives == 0:
        return 1.0
    return negatives / positives


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


def tune_xgboost(X_train, y_train, scale_pos_weight: float) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "aucpr",
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params


def tune_lightgbm(X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "is_unbalance": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost and LightGBM classifiers")
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

    scale_pos_weight = compute_scale_pos_weight(y_train)
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)

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
    xgb_best_params = load_best_params(args.params, "xgboost")
    if xgb_best_params is not None:
        print(f"\nLoaded pre-tuned XGBoost params from {args.params}")
        print(f"Params: {xgb_best_params}")
    else:
        print(f"\nTuning XGBoost ({N_TRIALS} Optuna trials, {CV_FOLDS}-fold CV)...")
        xgb_best_params = tune_xgboost(X_train, y_train, scale_pos_weight)
        print(f"Best XGBoost params: {xgb_best_params}")

    print(f"Selecting XGBoost decision threshold for target recall {args.target_recall:.2%}...")
    xgb_oof_probs = generate_oof_probabilities(
        X_train_raw,
        y_train,
        lambda fold_labels: XGBClassifier(
            **xgb_best_params,
            scale_pos_weight=compute_scale_pos_weight(fold_labels),
            random_state=42,
            n_jobs=-1,
            eval_metric="aucpr",
            verbosity=0,
        ),
    )
    xgb_selected_threshold, xgb_threshold_reason = select_threshold_for_recall(
        y_train,
        xgb_oof_probs,
        args.target_recall,
    )
    print(
        "Selected XGBoost threshold "
        f"{xgb_selected_threshold['decision_threshold']:.4f} "
        f"(OOF recall={xgb_selected_threshold['recall']:.4f}, "
        f"precision={xgb_selected_threshold['precision']:.4f}, "
        f"false_negatives={xgb_selected_threshold['false_negatives']})"
    )

    print("Training XGBoost with best params...")
    xgb_model = XGBClassifier(
        **xgb_best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr",
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_probability_metrics(
        y_test,
        xgb_probs,
        xgb_selected_threshold["decision_threshold"],
    )
    xgb_preds = (xgb_probs >= xgb_metrics["decision_threshold"]).astype(int)
    xgb_default_metrics = compute_probability_metrics(y_test, xgb_probs, 0.5)

    xgb_artifact_path = save_model_and_report(
        xgb_model,
        vectorizer,
        constant_columns,
        xgb_metrics["decision_threshold"],
        {
            **shared_meta,
            **xgb_metrics,
            "best_params": xgb_best_params,
            "default_threshold_metrics": xgb_default_metrics,
            "threshold_selection": {
                "selection_basis": "out_of_fold_training_predictions",
                "target_recall": args.target_recall,
                "selection_rule": xgb_threshold_reason,
                "selected_training_metrics": xgb_selected_threshold,
                "oof_threshold_sweep": build_threshold_sweep(
                    y_train,
                    xgb_oof_probs,
                    build_report_thresholds(xgb_selected_threshold["decision_threshold"]),
                ),
                "test_threshold_sweep": build_threshold_sweep(
                    y_test,
                    xgb_probs,
                    build_report_thresholds(xgb_selected_threshold["decision_threshold"]),
                ),
            },
        },
        BASE_DIR / "models" / "xgboost_model.joblib",
        BASE_DIR / "reports" / "xgboost_metrics.json",
        "xgboost",
    )

    print("\nXGBoost classification report (selected threshold):")
    print(classification_report(
        y_test, xgb_preds, digits=4, zero_division=0,
        target_names=["not_malicious", "malicious"],
    ))

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    lgbm_best_params = load_best_params(args.params, "lightgbm")
    if lgbm_best_params is not None:
        print(f"\nLoaded pre-tuned LightGBM params from {args.params}")
        print(f"Params: {lgbm_best_params}")
    else:
        print(f"\nTuning LightGBM ({N_TRIALS} Optuna trials, {CV_FOLDS}-fold CV)...")
        lgbm_best_params = tune_lightgbm(X_train, y_train)
        print(f"Best LightGBM params: {lgbm_best_params}")

    print(f"Selecting LightGBM decision threshold for target recall {args.target_recall:.2%}...")
    lgbm_oof_probs = generate_oof_probabilities(
        X_train_raw,
        y_train,
        lambda _: LGBMClassifier(
            **lgbm_best_params,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    )
    lgbm_selected_threshold, lgbm_threshold_reason = select_threshold_for_recall(
        y_train,
        lgbm_oof_probs,
        args.target_recall,
    )
    print(
        "Selected LightGBM threshold "
        f"{lgbm_selected_threshold['decision_threshold']:.4f} "
        f"(OOF recall={lgbm_selected_threshold['recall']:.4f}, "
        f"precision={lgbm_selected_threshold['precision']:.4f}, "
        f"false_negatives={lgbm_selected_threshold['false_negatives']})"
    )

    print("Training LightGBM with best params...")
    lgbm_model = LGBMClassifier(
        **lgbm_best_params,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_metrics = compute_probability_metrics(
        y_test,
        lgbm_probs,
        lgbm_selected_threshold["decision_threshold"],
    )
    lgbm_preds = (lgbm_probs >= lgbm_metrics["decision_threshold"]).astype(int)
    lgbm_default_metrics = compute_probability_metrics(y_test, lgbm_probs, 0.5)

    lgbm_artifact_path = save_model_and_report(
        lgbm_model,
        vectorizer,
        constant_columns,
        lgbm_metrics["decision_threshold"],
        {
            **shared_meta,
            **lgbm_metrics,
            "best_params": lgbm_best_params,
            "default_threshold_metrics": lgbm_default_metrics,
            "threshold_selection": {
                "selection_basis": "out_of_fold_training_predictions",
                "target_recall": args.target_recall,
                "selection_rule": lgbm_threshold_reason,
                "selected_training_metrics": lgbm_selected_threshold,
                "oof_threshold_sweep": build_threshold_sweep(
                    y_train,
                    lgbm_oof_probs,
                    build_report_thresholds(lgbm_selected_threshold["decision_threshold"]),
                ),
                "test_threshold_sweep": build_threshold_sweep(
                    y_test,
                    lgbm_probs,
                    build_report_thresholds(lgbm_selected_threshold["decision_threshold"]),
                ),
            },
        },
        BASE_DIR / "models" / "lightgbm_model.joblib",
        BASE_DIR / "reports" / "lightgbm_metrics.json",
        "lightgbm",
    )

    print("\nLightGBM classification report (selected threshold):")
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

    for metric in ["decision_threshold", "false_negatives", "false_positives", "roc_auc", "pr_auc", "precision", "recall", "f1"]:
        xgb_val = xgb_metrics[metric]
        lgbm_val = lgbm_metrics[metric]
        if has_rf:
            rf_val = rf_metrics.get(metric)
            print(
                f"{metric:<20} "
                f"{format_metric_value(metric, rf_val):>14} "
                f"{format_metric_value(metric, xgb_val):>10} "
                f"{format_metric_value(metric, lgbm_val):>10}"
            )
        else:
            print(
                f"{metric:<20} "
                f"{format_metric_value(metric, xgb_val):>10} "
                f"{format_metric_value(metric, lgbm_val):>10}"
            )

    print("=" * 65)
    print(f"\nXGBoost model  -> models/xgboost_model.joblib")
    print(f"LightGBM model -> models/lightgbm_model.joblib")
    print(f"XGBoost backend artifact  -> {xgb_artifact_path}")
    print(f"LightGBM backend artifact -> {lgbm_artifact_path}")


if __name__ == "__main__":
    main()
