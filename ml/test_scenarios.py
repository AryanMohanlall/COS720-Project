#!/usr/bin/env python3
"""
Scenario testing tool for insider threat detection models.

Usage examples
--------------
  python ml/test_scenarios.py --n 100 --model xgboost
  python ml/test_scenarios.py --n 200 --all-models --seed 7
  python ml/test_scenarios.py --n 50 --no-stratify
  python ml/test_scenarios.py --n 100 --output results.md
"""

import argparse
import csv
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "insider_threat_clean_dataset.csv"
MODEL_PATHS = {
    "xgboost":       BASE_DIR / "models" / "xgboost_model.joblib",
    "lightgbm":      BASE_DIR / "models" / "lightgbm_model.joblib",
    "random_forest": BASE_DIR / "models" / "random_forest_model.joblib",
}
TARGET_COLUMN = "is_malicious"
NUMERIC_COLUMNS = {
    "employee_seniority_years", "is_contractor", "employee_classification",
    "has_foreign_citizenship", "has_criminal_record", "has_medical_history",
    "total_printed_pages", "num_printed_pages_off_hours", "total_files_burned",
    "burned_from_other", "is_abroad", "trip_day_number", "hostility_country_level",
    "num_entries", "num_unique_campus", "late_exit_flag", "entry_during_weekend",
}

W = 82  # report width


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    with DATASET_PATH.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def parse_feature(column: str, value: str):
    if value is None or str(value).strip() == "":
        return None
    if column in NUMERIC_COLUMNS:
        v = float(str(value).strip())
        return int(v) if v == int(v) else v
    return str(value).strip()


def row_to_features(row: dict, constant_columns: set) -> dict:
    return {
        col: parse_feature(col, val)
        for col, val in row.items()
        if col != TARGET_COLUMN and col not in constant_columns
    }


def sample_records(rows: list[dict], n: int, stratified: bool) -> list[dict]:
    if stratified:
        positives = [r for r in rows if r[TARGET_COLUMN] == "1"]
        negatives = [r for r in rows if r[TARGET_COLUMN] == "0"]
        n_pos = max(1, n // 4)
        n_neg = n - n_pos
        sample = (
            random.sample(positives, min(n_pos, len(positives)))
            + random.sample(negatives, min(n_neg, len(negatives)))
        )
        random.shuffle(sample)
        return sample
    return random.sample(rows, min(n, len(rows)))


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(bundle: dict, features: dict) -> tuple[int, float, float]:
    """Returns (prediction, probability, confidence)."""
    vectorized = bundle["vectorizer"].transform([features])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        prob = float(bundle["model"].predict_proba(vectorized)[0, 1])
    threshold = float(bundle["decision_threshold"])
    return int(prob >= threshold), prob, max(prob, 1.0 - prob)


def classify(pred: int, actual: int) -> str:
    if pred == 1 and actual == 1: return "TP"
    if pred == 0 and actual == 0: return "TN"
    if pred == 1 and actual == 0: return "FP"
    return "FN"


def run_batch(bundle: dict, rows: list[dict]) -> list[dict]:
    constant_columns = set(bundle.get("constant_columns", []))
    results = []
    for row in rows:
        actual = int(float(row[TARGET_COLUMN]))
        features = row_to_features(row, constant_columns)
        pred, prob, conf = predict(bundle, features)
        results.append({
            "actual": actual,
            "predicted": pred,
            "probability": round(prob, 4),
            "confidence": round(conf, 4),
            "threshold": round(float(bundle["decision_threshold"]), 4),
            "outcome": classify(pred, actual),
            "features": features,
        })
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _sep(char="-"):
    print(char * W)


def confusion_summary(results: list[dict]) -> dict[str, int]:
    counts = Counter(r["outcome"] for r in results)
    return {
        "true_positives": counts["TP"],
        "true_negatives": counts["TN"],
        "false_positives": counts["FP"],
        "false_negatives": counts["FN"],
    }


def print_report(model_name: str, results: list[dict]):
    summary = confusion_summary(results)

    _sep("=")
    print(f"  SCENARIO TEST SUMMARY  -  {model_name.upper()}")
    _sep("=")
    print()
    print(f"  True positives  : {summary['true_positives']:>5}")
    print(f"  True negatives  : {summary['true_negatives']:>5}")
    print(f"  False positives : {summary['false_positives']:>5}")
    print(f"  False negatives : {summary['false_negatives']:>5}")
    print()


# ── Export ────────────────────────────────────────────────────────────────────

def _summary_row(results: list[dict], model_name: str) -> dict:
    summary = confusion_summary(results)
    return {
        "model": model_name,
        "true_positives": summary["true_positives"],
        "true_negatives": summary["true_negatives"],
        "false_positives": summary["false_positives"],
        "false_negatives": summary["false_negatives"],
    }


def export_csv(results: list[dict], model_name: str, output_path: Path):
    fieldnames = ["model", "true_positives", "true_negatives", "false_positives", "false_negatives"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_summary_row(results, model_name))
    print(f"  Results exported -> {output_path}")


def export_markdown(results: list[dict], model_name: str, output_path: Path):
    row = _summary_row(results, model_name)
    lines = [
        "# Scenario Test Summary",
        "",
        "| Model | True positives | True negatives | False positives | False negatives |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {row['model']} | {row['true_positives']} | "
            f"{row['true_negatives']} | {row['false_positives']} | "
            f"{row['false_negatives']} |"
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Results exported -> {output_path}")


def export_results(results: list[dict], model_name: str, output_path: Path):
    suffix = output_path.suffix.lower()
    if suffix == ".md":
        export_markdown(results, model_name, output_path)
    elif suffix == ".csv":
        export_csv(results, model_name, output_path)
    else:
        raise ValueError("Unsupported output format. Use .md or .csv")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Insider threat model scenario tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", choices=list(MODEL_PATHS), default="xgboost",
                        help="Which model to use (default: xgboost)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all three models on the same sample")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of records to sample (random mode, default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Pure random sampling instead of stratified (may yield few positives)")
    parser.add_argument("--output", metavar="FILE",
                        help="Export TP/FP/FN summary to .csv or .md")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_names = list(MODEL_PATHS.keys()) if args.all_models else [args.model]

    # Load requested model bundles
    bundles: dict[str, dict] = {}
    for name in model_names:
        path = MODEL_PATHS[name]
        if not path.exists():
            print(f"  [WARN] Model not found: {path}  (run training scripts first)", file=sys.stderr)
            continue
        print(f"  Loading {name}...")
        bundles[name] = joblib.load(path)

    if not bundles:
        print("  No models loaded. Run the training scripts first.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading dataset ({DATASET_PATH.name})...")
    rows = load_dataset()
    sample = sample_records(rows, args.n, stratified=not args.no_stratify)
    n_pos = sum(1 for r in sample if r[TARGET_COLUMN] == "1")
    print(f"  Sampled {len(sample)} records - {n_pos} malicious, {len(sample) - n_pos} benign\n")

    for name, bundle in bundles.items():
        results = run_batch(bundle, sample)
        print_report(name, results)
        if args.output:
            suffix = f"_{name}" if len(bundles) > 1 else ""
            out_path = Path(args.output)
            out_path = out_path.with_stem(out_path.stem + suffix)
            try:
                export_results(results, name, out_path)
            except ValueError as exc:
                print(f"  [ERROR] {exc}", file=sys.stderr)
                sys.exit(2)


if __name__ == "__main__":
    main()
