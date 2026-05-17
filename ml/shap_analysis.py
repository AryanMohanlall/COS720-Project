"""
shap_analysis.py — Global SHAP explainability analysis for trained insider threat classifiers.

Produces (in ml/reports/):
    <model>_shap_bar.png        Mean |SHAP| per raw feature — global importance ranking
    <model>_shap_beeswarm.png   SHAP value distribution — direction and magnitude per feature
    <model>_shap_summary.json   Ranked per-feature SHAP statistics (mean, std, mean abs)

The DictVectorizer one-hot encodes categorical features, so encoded column SHAPs are summed
back to raw-feature level before plotting (e.g. all employee_department=* columns → one bar).

Usage:
    python shap_analysis.py --model xgboost
    python shap_analysis.py --model lightgbm
    python shap_analysis.py --model random_forest --sample 2000
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from training_common import BASE_DIR, DATASET_PATH, load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────

REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILENAMES = {
    "xgboost": "xgboost_model.joblib",
    "lightgbm": "lightgbm_model.joblib",
    "random_forest": "random_forest_model.joblib",
}

# Human-readable labels for plots and JSON output
FEATURE_LABELS: dict[str, str] = {
    "employee_department": "Department",
    "employee_campus": "Campus",
    "employee_position": "Position",
    "employee_seniority_years": "Seniority (yrs)",
    "is_contractor": "Contractor",
    "employee_classification": "Classification",
    "has_foreign_citizenship": "Foreign citizenship",
    "has_criminal_record": "Criminal record",
    "has_medical_history": "Medical history",
    "employee_origin_country": "Origin country",
    "total_printed_pages": "Pages printed",
    "num_printed_pages_off_hours": "Off-hours pages",
    "total_files_burned": "Files burned",
    "burned_from_other": "Burned (other user)",
    "is_abroad": "Currently abroad",
    "trip_day_number": "Trip day no.",
    "hostility_country_level": "Hostility level",
    "num_entries": "Access entries",
    "num_unique_campus": "Unique campuses",
    "late_exit_flag": "Late exit",
    "entry_during_weekend": "Weekend entry",
}


# ── Feature aggregation helpers ───────────────────────────────────────────────


def _raw_name(encoded_name: str) -> str:
    """Strip the one-hot suffix: 'employee_department=IT' → 'employee_department'."""
    return encoded_name.split("=", 1)[0] if "=" in encoded_name else encoded_name


def build_feature_groups(vectorizer) -> dict[str, list[int]]:
    """Map each raw feature name to the list of encoded-column indices it spans."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(vectorizer.get_feature_names_out()):
        groups.setdefault(_raw_name(name), []).append(i)
    return groups


def aggregate_shap_to_raw(
    sv: np.ndarray,
    groups: dict[str, list[int]],
) -> tuple[np.ndarray, list[str]]:
    """Sum encoded-column SHAP values into per-raw-feature SHAP values.

    Returns aggregated matrix (n_samples, n_raw_features) and the ordered feature names.
    Categorical features are represented by the algebraic sum of their one-hot column SHAPs,
    which preserves direction (positive → pushes toward malicious).
    """
    raw_features = sorted(groups.keys())
    agg = np.zeros((sv.shape[0], len(raw_features)), dtype=float)
    for j, feat in enumerate(raw_features):
        agg[:, j] = sv[:, groups[feat]].sum(axis=1)
    return agg, raw_features


def build_numeric_value_matrix(
    X_test_raw: list[dict],
    raw_features: list[str],
) -> np.ndarray:
    """Extract numeric feature values for beeswarm colour coding.

    Categorical features are left as 0.0 (no meaningful ordinal ordering).
    """
    matrix = np.zeros((len(X_test_raw), len(raw_features)), dtype=float)
    for j, feat in enumerate(raw_features):
        for i, row in enumerate(X_test_raw):
            val = row.get(feat)
            if isinstance(val, (int, float)) and val is not None:
                matrix[i, j] = float(val)
    return matrix


def extract_shap_array(sv_raw) -> np.ndarray:
    """Normalise the various shapes TreeExplainer can return into (n_samples, n_features)."""
    if isinstance(sv_raw, list):
        # Older shap / Random Forest: [class_0_array, class_1_array]
        return np.asarray(sv_raw[1] if len(sv_raw) == 2 else sv_raw[0])
    arr = np.asarray(sv_raw)
    if arr.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — take positive class
        return arr[:, :, 1]
    # Shape (n_samples, n_features) — already positive class
    return arr


def compute_shap_values(model, X_test: np.ndarray) -> np.ndarray:
    """Compute SHAP values, dispatching to model-native methods where available.

    XGBoost 2.x stores base_score in a format (e.g. '[5E-1]') that shap's
    TreeExplainer cannot parse. Both XGBoost and LightGBM expose their own
    SHAP computation via pred_contribs on the underlying booster, which is
    both faster and free of the version-compatibility issue.

    Random Forest has no such native method, so shap.TreeExplainer is used there.

    Returns:
        np.ndarray of shape (n_samples, n_encoded_features).
        Positive values push the prediction toward malicious; negative away from it.
        XGBoost and LightGBM values are in log-odds space; Random Forest in probability space.
    """
    model_module = type(model).__module__.split(".")[0]

    if model_module == "xgboost":
        import xgboost as xgb

        # pred_contribs=True → (n_samples, n_features + 1); last column is the bias term
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dmat = xgb.DMatrix(X_test)
            sv_with_bias = model.get_booster().predict(dmat, pred_contribs=True)
        return np.asarray(sv_with_bias)[:, :-1]

    if model_module == "lightgbm":
        # booster_.predict with pred_contrib=True → (n_samples, n_features + 1)
        sv_with_bias = model.booster_.predict(X_test, pred_contrib=True)
        return np.asarray(sv_with_bias)[:, :-1]

    # Random Forest and other sklearn estimators — shap.TreeExplainer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(X_test, check_additivity=False)
    return extract_shap_array(sv_raw)


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_bar(
    mean_abs_shap: dict[str, float],
    model_label: str,
    out_path: Path,
) -> None:
    """Horizontal bar chart: mean |SHAP value| per raw feature, ranked descending."""
    ranked_feats = sorted(mean_abs_shap, key=mean_abs_shap.get, reverse=True)
    values = [mean_abs_shap[f] for f in ranked_feats]
    labels = [FEATURE_LABELS.get(f, f) for f in ranked_feats]

    fig, ax = plt.subplots(figsize=(8, max(5, len(ranked_feats) * 0.38)))
    ax.barh(range(len(ranked_feats)), values, color="#2563eb", edgecolor="none", height=0.65)
    ax.set_yticks(range(len(ranked_feats)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| — average impact on model output", fontsize=9)
    ax.set_title(
        f"{model_label} — Global Feature Importance (SHAP)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart  → {out_path}")


def plot_beeswarm(
    agg_shap: np.ndarray,
    feature_vals: np.ndarray,
    raw_features: list[str],
    mean_abs_shap: dict[str, float],
    model_label: str,
    out_path: Path,
) -> None:
    """Beeswarm plot: one dot per test sample per feature.

    Dots are coloured by normalised feature value (blue=low, red=high).
    Features are ordered top→bottom by mean |SHAP|.
    Positive x → pushes prediction toward malicious; negative → away.
    """
    order = sorted(
        range(len(raw_features)),
        key=lambda j: mean_abs_shap[raw_features[j]],
        reverse=True,
    )
    sorted_feats = [raw_features[j] for j in order]
    sorted_labels = [FEATURE_LABELS.get(f, f) for f in sorted_feats]
    sorted_shap = agg_shap[:, order]
    sorted_vals = feature_vals[:, order]

    n_features = len(sorted_feats)
    fig, ax = plt.subplots(figsize=(9, max(5, n_features * 0.42)))
    cmap = plt.get_cmap("RdBu_r")  # blue = low feature value, red = high

    rng = np.random.default_rng(42)
    for j in range(n_features):
        y_base = n_features - 1 - j  # top = most important
        shap_col = sorted_shap[:, j]
        val_col = sorted_vals[:, j]

        v_min, v_max = val_col.min(), val_col.max()
        normed = (
            (val_col - v_min) / (v_max - v_min)
            if v_max > v_min
            else np.full_like(val_col, 0.5)
        )

        y_jitter = rng.uniform(-0.32, 0.32, size=len(shap_col))
        ax.scatter(
            shap_col,
            y_base + y_jitter,
            c=cmap(normed),
            alpha=0.35,
            s=6,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(reversed(sorted_labels), fontsize=9)
    ax.set_xlabel("SHAP value  (positive → increases risk score)", fontsize=9)
    ax.set_title(
        f"{model_label} — SHAP Value Distribution per Feature",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.018, aspect=30)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.set_ylabel("Feature value", fontsize=7, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Beeswarm   → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute global SHAP explanations for a trained insider threat classifier "
            "and save bar chart, beeswarm plot, and JSON summary to ml/reports/."
        )
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_FILENAMES),
        default="xgboost",
        help="Which trained model to explain (default: xgboost)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override the model .joblib path (default: ml/models/<model>_model.joblib)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        metavar="PATH",
        help="Dataset CSV used during training (default: ml/data/insider_threat_clean_dataset.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Subsample N rows from the test set for SHAP computation. "
            "Recommended for Random Forest (e.g. --sample 2000). Default: use all test rows."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42)",
    )
    args = parser.parse_args()

    model_path = args.model_path or MODEL_DIR / MODEL_FILENAMES[args.model]
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    # ── Load model artifact ──────────────────────────────────────────────────
    print(f"\nLoading model: {model_path}")
    payload = joblib.load(model_path)
    model = payload["model"]
    vectorizer = payload["vectorizer"]
    model_name: str = payload.get("model_name", args.model)
    model_label = model_name.upper().replace("_", " ")

    # ── Recreate identical test split ────────────────────────────────────────
    print("Preparing test set (same split as training: test_size=0.2, random_state=42)...")
    features, labels, _ = load_dataset(args.dataset)
    _, X_test_raw, _, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_test = np.asarray(vectorizer.transform(X_test_raw))
    y_test_arr = np.asarray(y_test)

    # ── Optional subsampling ─────────────────────────────────────────────────
    if args.sample is not None and args.sample < len(X_test_raw):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_test_raw), size=args.sample, replace=False)
        X_test = X_test[idx]
        X_test_raw = [X_test_raw[i] for i in idx]
        y_test_arr = y_test_arr[idx]
        print(f"  Subsampled to {args.sample} rows (seed={args.seed})")

    n_positive = int(y_test_arr.sum())
    print(
        f"  Rows: {len(X_test_raw)}  |  "
        f"Malicious: {n_positive} ({n_positive / len(X_test_raw):.1%})  |  "
        f"Encoded features: {X_test.shape[1]}"
    )

    # ── Compute SHAP values ──────────────────────────────────────────────────
    print("Computing SHAP values...")
    sv = compute_shap_values(model, X_test)
    print(f"  SHAP matrix: {sv.shape}  (n_samples × n_encoded_features)")

    # ── Aggregate encoded columns → raw features ─────────────────────────────
    groups = build_feature_groups(vectorizer)
    agg_shap, raw_features = aggregate_shap_to_raw(sv, groups)
    feature_vals = build_numeric_value_matrix(X_test_raw, raw_features)

    feat_idx = {f: j for j, f in enumerate(raw_features)}
    mean_abs_shap = {f: float(np.abs(agg_shap[:, feat_idx[f]]).mean()) for f in raw_features}
    ranked = sorted(mean_abs_shap.items(), key=lambda x: x[1], reverse=True)

    print("\nGlobal SHAP importance — top 10 raw features (mean |SHAP|):")
    for feat, score in ranked[:10]:
        label = FEATURE_LABELS.get(feat, feat)
        print(f"  {label:<28}  {score:.5f}")

    # ── Save JSON summary ────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{model_name}_shap_summary.json"

    summary = {
        "model": model_name,
        "test_rows": len(X_test_raw),
        "positive_count": n_positive,
        "positive_rate": round(n_positive / len(X_test_raw), 4),
        "shap_explainer": "shap.TreeExplainer",
        "shap_output": (
            "log-odds space for gradient-boosted models; "
            "probability space for Random Forest"
        ),
        "aggregation": (
            "encoded columns for each raw feature are summed; "
            "sign is preserved (positive SHAP → pushes toward malicious)"
        ),
        "features_ranked": [
            {
                "rank": rank + 1,
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "encoded_columns": len(groups[feat]),
                "mean_abs_shap": round(score, 6),
                "mean_shap": round(float(np.mean(agg_shap[:, feat_idx[feat]])), 6),
                "std_shap": round(float(np.std(agg_shap[:, feat_idx[feat]])), 6),
                "max_shap": round(float(np.max(agg_shap[:, feat_idx[feat]])), 6),
                "min_shap": round(float(np.min(agg_shap[:, feat_idx[feat]])), 6),
            }
            for rank, (feat, score) in enumerate(ranked)
        ],
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  JSON summary → {json_path}")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("Generating plots...")
    plot_bar(
        mean_abs_shap,
        model_label,
        REPORTS_DIR / f"{model_name}_shap_bar.png",
    )
    plot_beeswarm(
        agg_shap,
        feature_vals,
        raw_features,
        mean_abs_shap,
        model_label,
        REPORTS_DIR / f"{model_name}_shap_beeswarm.png",
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
