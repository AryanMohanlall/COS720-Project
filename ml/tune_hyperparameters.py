"""
Hyperparameter tuning for Random Forest, XGBoost, and LightGBM using Optuna.

Runs N_TRIALS Bayesian trials per model with CV_FOLDS-fold stratified CV,
scoring on average_precision (PR-AUC). Saves best params per model to
reports/best_params.json.
"""
import csv
import json
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "best_params.json"
TARGET_COLUMN = "is_malicious"

N_TRIALS = 50
CV_FOLDS = 5
RF_TUNING_SAMPLE_SIZE = 30000

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
    features, labels, column_values = [], [], {}

    with dataset_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(float(row[TARGET_COLUMN]))
            feature_row = {}
            for col, raw in row.items():
                if col == TARGET_COLUMN:
                    continue
                parsed = parse_value(col, raw)
                feature_row[col] = parsed
                column_values.setdefault(col, set()).add(parsed)
            features.append(feature_row)
            labels.append(label)

    constant_columns = {col for col, vals in column_values.items() if len(vals) <= 1}
    if constant_columns:
        for feature_row in features:
            for col in constant_columns:
                feature_row.pop(col, None)

    return features, labels


def tune_xgboost(X_train: pd.DataFrame, y_train, scale_pos_weight: float) -> tuple[dict, float]:
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
            "n_jobs": 1,
            "eval_metric": "aucpr",
            "verbosity": 0,
        }
        scores = cross_val_score(XGBClassifier(**params), X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params, study.best_value


def tune_lightgbm(X_train: pd.DataFrame, y_train) -> tuple[dict, float]:
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
            "n_jobs": 1,
            "verbose": -1,
        }
        scores = cross_val_score(LGBMClassifier(**params), X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params, study.best_value


def tune_random_forest(X_train: pd.DataFrame, y_train) -> tuple[dict, float]:
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
            # Parallelize tree building inside each fit and keep CV serial to avoid nested oversubscription.
            "n_jobs": -1,
        }
        scores = cross_val_score(RandomForestClassifier(**params), X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    return study.best_params, study.best_value


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading dataset...")
    features, labels = load_dataset(DATASET_PATH)

    X_train_raw, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels,
    )

    vectorizer = DictVectorizer(sparse=False)
    X_train = pd.DataFrame(
        vectorizer.fit_transform(X_train_raw),
        columns=vectorizer.get_feature_names_out(),
    )

    neg = sum(1 for y in y_train if y == 0)
    pos = sum(1 for y in y_train if y == 1)
    scale_pos_weight = neg / pos

    results = {}
    X_train_rf = X_train
    y_train_rf = y_train

    if RF_TUNING_SAMPLE_SIZE and RF_TUNING_SAMPLE_SIZE < len(y_train):
        X_train_rf, _, y_train_rf, _ = train_test_split(
            X_train,
            y_train,
            train_size=RF_TUNING_SAMPLE_SIZE,
            random_state=42,
            stratify=y_train,
        )
        print(
            f"Random Forest tuning will use a stratified subsample of "
            f"{len(y_train_rf):,} rows out of {len(y_train):,} training rows."
        )

    print(f"\nTuning XGBoost ({N_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    xgb_params, xgb_score = tune_xgboost(X_train, y_train, scale_pos_weight)
    results["xgboost"] = {"best_params": xgb_params, "cv_pr_auc": round(xgb_score, 6)}
    print(f"  Best CV PR-AUC: {xgb_score:.4f}")
    print(f"  Params: {xgb_params}")

    print(f"\nTuning LightGBM ({N_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    lgbm_params, lgbm_score = tune_lightgbm(X_train, y_train)
    results["lightgbm"] = {"best_params": lgbm_params, "cv_pr_auc": round(lgbm_score, 6)}
    print(f"  Best CV PR-AUC: {lgbm_score:.4f}")
    print(f"  Params: {lgbm_params}")

    print(f"\nTuning Random Forest ({N_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    rf_params, rf_score = tune_random_forest(X_train_rf, y_train_rf)
    results["random_forest"] = {"best_params": rf_params, "cv_pr_auc": round(rf_score, 6)}
    print(f"  Best CV PR-AUC: {rf_score:.4f}")
    print(f"  Params: {rf_params}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("TUNING SUMMARY")
    print("=" * 50)
    print(f"{'Model':<20} {'CV PR-AUC':>10}")
    print("-" * 32)
    for model, data in results.items():
        print(f"{model:<20} {data['cv_pr_auc']:>10.4f}")
    print("=" * 50)
    print(f"\nBest params saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
