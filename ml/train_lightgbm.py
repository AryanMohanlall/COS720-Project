import argparse
from pathlib import Path

import optuna
from lightgbm import LGBMClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from training_common import (
    BASE_DIR,
    BEST_PARAMS_PATH,
    CV_FOLDS,
    DATASET_PATH,
    DEFAULT_TARGET_RECALL,
    N_TRIALS,
    build_report_thresholds,
    build_shared_meta,
    build_threshold_sweep,
    compute_probability_metrics,
    generate_oof_probabilities,
    load_best_params,
    load_dataset,
    predict_positive_probabilities,
    save_model_and_report,
    select_threshold_for_recall,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    parser = argparse.ArgumentParser(description="Train a LightGBM insider threat classifier")
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

    shared_meta = build_shared_meta(
        DATASET_PATH,
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        X_train,
        constant_columns,
    )

    best_params = load_best_params(args.params, "lightgbm")
    if best_params is not None:
        print(f"\nLoaded pre-tuned LightGBM params from {args.params}")
        print(f"Params: {best_params}")
    else:
        print(f"\nTuning LightGBM ({N_TRIALS} Optuna trials, {CV_FOLDS}-fold CV)...")
        best_params = tune_lightgbm(X_train, y_train)
        print(f"Best LightGBM params: {best_params}")

    print(f"Selecting LightGBM decision threshold for target recall {args.target_recall:.2%}...")
    oof_probs = generate_oof_probabilities(
        X_train_raw,
        y_train,
        lambda _: LGBMClassifier(
            **best_params,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    )
    selected_threshold, threshold_reason = select_threshold_for_recall(
        y_train,
        oof_probs,
        args.target_recall,
    )
    print(
        "Selected LightGBM threshold "
        f"{selected_threshold['decision_threshold']:.4f} "
        f"(OOF recall={selected_threshold['recall']:.4f}, "
        f"precision={selected_threshold['precision']:.4f}, "
        f"false_negatives={selected_threshold['false_negatives']})"
    )

    print("Training LightGBM with best params...")
    model = LGBMClassifier(
        **best_params,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    probabilities = predict_positive_probabilities(model, X_test)
    metrics = compute_probability_metrics(
        y_test,
        probabilities,
        selected_threshold["decision_threshold"],
    )
    predictions = (probabilities >= metrics["decision_threshold"]).astype(int)
    default_threshold_metrics = compute_probability_metrics(y_test, probabilities, 0.5)

    artifact_path = save_model_and_report(
        model,
        vectorizer,
        constant_columns,
        metrics["decision_threshold"],
        {
            **shared_meta,
            **metrics,
            "best_params": best_params,
            "default_threshold_metrics": default_threshold_metrics,
            "threshold_selection": {
                "selection_basis": "out_of_fold_training_predictions",
                "target_recall": args.target_recall,
                "selection_rule": threshold_reason,
                "selected_training_metrics": selected_threshold,
                "oof_threshold_sweep": build_threshold_sweep(
                    y_train,
                    oof_probs,
                    build_report_thresholds(selected_threshold["decision_threshold"]),
                ),
                "test_threshold_sweep": build_threshold_sweep(
                    y_test,
                    probabilities,
                    build_report_thresholds(selected_threshold["decision_threshold"]),
                ),
            },
        },
        BASE_DIR / "models" / "lightgbm_model.joblib",
        BASE_DIR / "reports" / "lightgbm_metrics.json",
        "lightgbm",
    )

    print("\nLightGBM classification report (selected threshold):")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
            target_names=["not_malicious", "malicious"],
        )
    )
    print(f"Model saved to: {BASE_DIR / 'models' / 'lightgbm_model.joblib'}")
    print(f"Backend artifact saved to: {artifact_path}")
    print(f"Metrics saved to: {BASE_DIR / 'reports' / 'lightgbm_metrics.json'}")


if __name__ == "__main__":
    main()
