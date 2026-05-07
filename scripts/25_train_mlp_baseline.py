"""Step 17 -- MLP baseline classifier (Week 6).

Two-layer sklearn MLP as comparison baseline for the LightGBM Tier-A
model.  Demonstrates that LightGBM outperforms MLP in this small-data /
high-dimension regime (N=572, D=1953), supporting Paper 2 Method Section 3.

Architecture: (1953) -> 256 -> 128 -> 3 classes, ReLU, L2 reg=1e-4,
Adam, lr=1e-3, max_iter=200, early stopping patience=20.

Same 5-fold stratified CV and 1000x bootstrap CI as script 20.

Input:
  /data/features/fused/feature_matrix.parquet
  /data/features/fused/labels.parquet

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "git config --global --add safe.directory /pkg && \\
                 git config --global --add safe.directory /genome-atlas && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps && \\
                 pip install lightgbm scikit-learn --quiet && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --quiet && \\
                 python scripts/25_train_mlp_baseline.py"

Expected output:
  /data/models/mlp_baseline/mlp_cv_summary.json
  /data/models/mlp_baseline/mlp_cv_metrics.parquet
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEAT_PATH  = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH = Path("/data/features/fused/labels.parquet")
OUT_DIR    = Path("/data/models/mlp_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACC_COL = "uniprot_acc"
N_FOLDS = 5
N_BOOT  = 1000
SEED    = 42
CLASSES = ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE", "TRANSPOSASE"]

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    alpha=1e-4,          # L2 weight decay
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=SEED,
    verbose=False,
)


def bootstrap_macro_f1(y_true, y_pred, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        scores.append(float(f1_score(y_true[idx], y_pred[idx],
                                     average="macro", zero_division=0)))
    f1_pt = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return f1_pt, float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def run():
    feat_df  = pd.read_parquet(FEAT_PATH)
    label_df = pd.read_parquet(LABEL_PATH)

    merged = feat_df.merge(label_df[[ACC_COL, "tier_a"]], on=ACC_COL, how="inner")
    merged = merged.dropna(subset=["tier_a"])
    merged = merged[merged["tier_a"].isin(CLASSES)].reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c not in (ACC_COL, "tier_a")]
    X = merged[feature_cols].values.astype(np.float32)

    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(merged["tier_a"].values)

    n_samples = len(X)
    class_dist = {CLASSES[i]: int((y == i).sum()) for i in range(3)}
    print(f"Training samples : {n_samples}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {class_dist}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # StandardScaler per fold (MLP is sensitive to feature scale)
        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = MLPClassifier(**MLP_PARAMS)
        clf.fit(X_tr_s, y_tr)

        y_pred = clf.predict(X_val_s)
        f1_pt, f1_lo, f1_hi = bootstrap_macro_f1(y_val, y_pred)

        fold_records.append({
            "fold":     fold_idx,
            "n_train":  len(train_idx),
            "n_val":    len(val_idx),
            "macro_f1": f1_pt,
            "f1_lo":    f1_lo,
            "f1_hi":    f1_hi,
        })
        print(f"  Fold {fold_idx}: macro-F1 = {f1_pt:.4f}  [{f1_lo:.4f}, {f1_hi:.4f}]")

    cv_df = pd.DataFrame(fold_records)
    mean_f1 = float(cv_df["macro_f1"].mean())
    mean_lo = float(cv_df["f1_lo"].mean())
    mean_hi = float(cv_df["f1_hi"].mean())

    print(f"\n5-fold CV summary (MLP):")
    print(f"  mean macro-F1 : {mean_f1:.4f}")
    print(f"  mean CI lower : {mean_lo:.4f}")
    print(f"  mean CI upper : {mean_hi:.4f}")

    # Retrain final model on full dataset with StandardScaler
    print("\nRetraining on full dataset...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_clf = MLPClassifier(**MLP_PARAMS)
    final_clf.fit(X_scaled, y)

    # Save
    model_path = OUT_DIR / "model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model": final_clf,
            "scaler": final_scaler,
            "label_encoder": le,
            "feature_cols": feature_cols,
        }, fh)
    print(f"Model saved -> {model_path}")

    cv_path = OUT_DIR / "mlp_cv_metrics.parquet"
    cv_df.to_parquet(cv_path, compression="zstd")

    summary = {
        "mean_macro_f1":   mean_f1,
        "mean_f1_lo_2.5":  mean_lo,
        "mean_f1_hi_97.5": mean_hi,
        "n_train":         n_samples,
        "class_distribution": class_dist,
        "n_features":      X.shape[1],
        "n_folds":         N_FOLDS,
        "n_bootstrap":     N_BOOT,
        "fold_results":    fold_records,
        "model_params":    str(MLP_PARAMS),
    }
    summary_path = OUT_DIR / "mlp_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"CV summary -> {summary_path}")

    print("\n=== MLP Baseline Complete ===")
    print(f"  mean macro-F1: {mean_f1:.4f} [{mean_lo:.4f}, {mean_hi:.4f}]")
    print(f"  (Compare: LightGBM Tier-A 5-fold mean F1 from cv_summary.json)")


if __name__ == "__main__":
    run()
