#!/usr/bin/env python3
"""
Scenario testing tool for insider threat detection models.

Modes
-----
  random       Sample records from the dataset (stratified by default).
  interactive  Enter feature values manually at the terminal.

Usage examples
--------------
  python ml/test_scenarios.py --mode random --n 100 --model xgboost
  python ml/test_scenarios.py --mode random --n 200 --all-models --seed 7
  python ml/test_scenarios.py --mode random --n 50 --no-stratify
  python ml/test_scenarios.py --mode interactive --model lightgbm
  python ml/test_scenarios.py --mode random --n 100 --output results.md
  python ml/test_scenarios.py --mode random --n 100 --output results.png
"""

import argparse
import csv
import random
import struct
import sys
import warnings
import zlib
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
INTERACTIVE_DEFAULTS = {
    "employee_department": "IT",
    "employee_campus": "Campus A",
    "employee_position": "Analyst",
    "employee_seniority_years": 3,
    "is_contractor": 0,
    "employee_classification": 2,
    "has_foreign_citizenship": 0,
    "has_criminal_record": 0,
    "has_medical_history": 0,
    "employee_origin_country": "South Africa",
    "total_printed_pages": 0,
    "num_printed_pages_off_hours": 0,
    "total_files_burned": 0,
    "burned_from_other": 0,
    "is_abroad": 0,
    "trip_day_number": 0,
    "hostility_country_level": 1,
    "num_entries": 4,
    "num_unique_campus": 1,
    "late_exit_flag": 0,
    "entry_during_weekend": 0,
}

# Feature flags used in pattern analysis
_RISK_FLAGS = [
    ("total_files_burned",         lambda v: v > 0,    "burned files"),
    ("burned_from_other",          lambda v: v > 0,    "burned files from others"),
    ("has_criminal_record",        lambda v: v > 0,    "criminal record"),
    ("has_foreign_citizenship",    lambda v: v > 0,    "foreign citizenship"),
    ("is_abroad",                  lambda v: v > 0,    "abroad during trip"),
    ("late_exit_flag",             lambda v: v > 0,    "late building exits"),
    ("entry_during_weekend",       lambda v: v > 0,    "weekend entry"),
    ("num_printed_pages_off_hours",lambda v: v > 0,    "off-hours printing"),
    ("total_printed_pages",        lambda v: v > 100,  "high print volume (>100)"),
    ("hostility_country_level",    lambda v: v >= 3,   "high-hostility country (>=3)"),
    ("employee_classification",    lambda v: v >= 3,   "high clearance level (>=3)"),
    ("num_entries",                lambda v: v > 20,   "many building entries (>20)"),
]

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


def _risk_flags(features: dict) -> list[str]:
    return [
        label for feat, check, label in _RISK_FLAGS
        if check(features.get(feat) or 0)
    ]


def confusion_summary(results: list[dict]) -> dict[str, int]:
    counts = Counter(r["outcome"] for r in results)
    return {
        "true_positives": counts["TP"],
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
    print(f"  False positives : {summary['false_positives']:>5}")
    print(f"  False negatives : {summary['false_negatives']:>5}")
    print()


# ── Interactive mode ───────────────────────────────────────────────────────────

def interactive_mode(bundle: dict):
    constant_columns = set(bundle.get("constant_columns", []))
    features = {k: v for k, v in INTERACTIVE_DEFAULTS.items() if k not in constant_columns}
    threshold = float(bundle["decision_threshold"])

    print(f"\n  Model     : {bundle.get('model_name', '?')}")
    print(f"  Threshold : {threshold:.4f}")
    print("\n  Commands:")
    print("    <feature>=<value>   set a feature value")
    print("    list                show all current feature values")
    print("    predict             run prediction (will ask for ground-truth label)")
    print("    reset               reset to default values")
    print("    quit                exit\n")

    session_results = []

    while True:
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd.lower() in ("quit", "exit", "q"):
            break

        if cmd.lower() == "list":
            print()
            for k, v in sorted(features.items()):
                print(f"    {k:<35} = {v}")
            print()
            continue

        if cmd.lower() == "reset":
            features = {k: v for k, v in INTERACTIVE_DEFAULTS.items() if k not in constant_columns}
            print("  Features reset to defaults.\n")
            continue

        if cmd.lower() == "predict":
            pred, prob, conf = predict(bundle, features)
            label_str = "MALICIOUS" if pred else "CLEAR"
            print("\n  PREDICTION")
            print(f"  Result     : {label_str}")
            print(f"  Probability: {prob:.4f}  (threshold: {threshold:.4f})")
            print(f"  Confidence : {conf:.1%}")
            flags = _risk_flags(features)
            if flags:
                print(f"  Risk flags : {', '.join(flags)}")
            print()

            actual_str = input("  Actual label (0=benign / 1=malicious / blank=unknown): ").strip()
            if actual_str in ("0", "1"):
                actual = int(actual_str)
                outcome = classify(pred, actual)
                print(f"  Outcome: {outcome}\n")
                session_results.append({
                    "outcome": outcome, "predicted": pred, "actual": actual,
                    "probability": round(prob, 4), "confidence": round(conf, 4),
                    "threshold": round(threshold, 4), "features": dict(features),
                })
            else:
                print()
            continue

        # Parse feature=value
        if "=" in cmd:
            key, _, val = cmd.partition("=")
            key = key.strip()
            if key in constant_columns:
                print(f"  '{key}' is a constant column - the model ignores it.")
            else:
                features[key] = parse_feature(key, val.strip())
                print(f"  Set {key} = {features[key]}")
            continue

        print("  Unknown command. Type 'list', '<feature>=<value>', 'predict', or 'quit'.")

    # End-of-session summary
    if session_results:
        print()
        _sep("=")
        print("  INTERACTIVE SESSION SUMMARY")
        _sep("=")
        summary = confusion_summary(session_results)
        print(f"  True positives  : {summary['true_positives']:>5}")
        print(f"  False positives : {summary['false_positives']:>5}")
        print(f"  False negatives : {summary['false_negatives']:>5}")


# ── Export ────────────────────────────────────────────────────────────────────

def _summary_row(results: list[dict], model_name: str) -> dict:
    summary = confusion_summary(results)
    return {
        "model": model_name,
        "true_positives": summary["true_positives"],
        "false_positives": summary["false_positives"],
        "false_negatives": summary["false_negatives"],
    }


def export_csv(results: list[dict], model_name: str, output_path: Path):
    fieldnames = ["model", "true_positives", "false_positives", "false_negatives"]
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
        "| Model | True positives | False positives | False negatives |",
        "|---|---:|---:|---:|",
        (
            f"| {row['model']} | {row['true_positives']} | "
            f"{row['false_positives']} | {row['false_negatives']} |"
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Results exported -> {output_path}")


_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _fill_rect(pixels: bytearray, width: int, x: int, y: int, w: int, h: int, color: tuple[int, int, int]):
    for py in range(max(0, y), min(y + h, len(pixels) // (width * 3))):
        row = py * width * 3
        for px in range(max(0, x), min(x + w, width)):
            idx = row + px * 3
            pixels[idx:idx + 3] = bytes(color)


def _draw_text(
    pixels: bytearray,
    width: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: int = 3,
):
    cursor = x
    for char in text.upper():
        glyph = _FONT_5X7.get(char, _FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    _fill_rect(pixels, width, cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += 6 * scale


def _write_png(output_path: Path, width: int, height: int, pixels: bytearray):
    raw = b"".join(
        b"\x00" + pixels[y * width * 3:(y + 1) * width * 3]
        for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    output_path.write_bytes(png)


def export_png(results: list[dict], model_name: str, output_path: Path):
    row = _summary_row(results, model_name)
    labels = ["True positives", "False positives", "False negatives"]
    values = [row["true_positives"], row["false_positives"], row["false_negatives"]]

    width, height = 1100, 360
    pixels = bytearray([255, 255, 255] * width * height)
    navy = (42, 66, 87)
    border = (70, 70, 70)
    light = (247, 249, 251)
    text = (25, 25, 25)

    _draw_text(pixels, width, 60, 38, "Scenario Test Summary", navy, scale=4)
    table_x, table_y = 60, 115
    col_w = [260, 260, 260, 260]
    row_h = 72

    x = table_x
    for i, header in enumerate(["Model", *labels]):
        _fill_rect(pixels, width, x, table_y, col_w[i], row_h, navy)
        _fill_rect(pixels, width, x, table_y + row_h, col_w[i], row_h, light)
        _draw_text(pixels, width, x + 22, table_y + 25, header, (255, 255, 255), scale=2)
        x += col_w[i]

    x = table_x
    for i, value in enumerate([model_name, *[str(v) for v in values]]):
        _draw_text(pixels, width, x + 22, table_y + row_h + 25, value, text, scale=2)
        x += col_w[i]

    total_w = sum(col_w)
    for line_y in (table_y, table_y + row_h, table_y + row_h * 2):
        _fill_rect(pixels, width, table_x, line_y, total_w, 2, border)
    x = table_x
    for col in col_w:
        _fill_rect(pixels, width, x, table_y, 2, row_h * 2, border)
        x += col
    _fill_rect(pixels, width, x, table_y, 2, row_h * 2, border)

    _write_png(output_path, width, height, pixels)
    print(f"  Results exported -> {output_path}")


def export_results(results: list[dict], model_name: str, output_path: Path):
    suffix = output_path.suffix.lower()
    if suffix == ".md":
        export_markdown(results, model_name, output_path)
    elif suffix == ".png":
        export_png(results, model_name, output_path)
    elif suffix == ".csv":
        export_csv(results, model_name, output_path)
    else:
        raise ValueError("Unsupported output format. Use .md, .png, or .csv")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Insider threat model scenario tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["random", "interactive"], default="random",
                        help="random: sample from dataset  |  interactive: manual entry")
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
                        help="Export TP/FP/FN summary to .csv, .md, or .png (random mode only)")
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

    if args.mode == "interactive":
        name, bundle = next(iter(bundles.items()))
        if len(bundles) > 1:
            print(f"  [INFO] Interactive mode uses one model at a time - using {name}.")
        interactive_mode(bundle)
        return

    # Random mode
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
