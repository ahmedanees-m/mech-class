"""Train Tier-A LightGBM classifier.

Trains a 3-class LightGBM classifier:
  DSB_NUCLEASE | DSB_FREE_TRANSEST_RECOMBINASE | TRANSPOSASE

Uses stratified 5-fold CV with per-fold 1000x bootstrap CI (95%, 2.5th pct).
Gate 2 pre-registered criterion (S0.5):
  mean(macro_f1) >= 0.80  AND  mean(macro_f1_lo) >= 0.70
  where macro_f1_lo = 2.5th percentile of fold-level bootstrap distribution.

Input:
  /data/features/fused/feature_matrix.parquet   columns: uniprot_acc, feat_*
  /data/features/fused/labels.parquet           columns: uniprot_acc, tier_a, tier_b, ...

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install lightgbm scikit-learn --quiet && \\
                 pip install -e . --quiet && \\
                 python scripts/20_train_tier_a.py"

Expected output:
  /data/models/tier_a/model.pkl
  /data/models/tier_a/tier_a_cv_metrics.parquet
  /data/models/tier_a/feature_importance.parquet
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ?? Paths ?????????????????????????????????????????????????????????????????????
FEAT_PATH  = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH = Path("/data/features/fused/labels.parquet")
MODEL_DIR  = Path("/data/models/tier_a")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ?? Constants ?????????????????????????????????????????????????????????????????
ACC_COL  = "uniprot_acc"
N_FOLDS  = 5
N_BOOT   = 1000
SEED     = 42
CLASSES  = ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE", "TRANSPOSASE"]

# LightGBM hyperparameters (tuned for small-data / high-dim regime)
LGB_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      400,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 5,
    "colsample_bytree":  0.5,   # critical: 1953 features -> subsample cols
    "subsample":         0.8,
    "subsample_freq":    1,
    "reg_lambda":        0.1,
    "class_weight":      "balanced",   # DSB_NUCLEASE ~7% of training set
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}


# ?? Bootstrap helper ??????????????????????????????????????????????????????????

def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return (point_f1, ci_lo_2.5pct, ci_hi_97.5pct) via 1000x bootstrap."""
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        scores.append(
            float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
        )
    f1_pt = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return f1_pt, float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


# ?? Main ??????????????????????????????????????????????????????????????????????

def run() -> None:
    # 1. Load features and labels
    feat_df  = pd.read_parquet(FEAT_PATH)
    label_df = pd.read_parquet(LABEL_PATH)

    merged = feat_df.merge(label_df[[ACC_COL, "tier_a"]], on=ACC_COL, how="inner")
    merged = merged.dropna(subset=["tier_a"])

    # Exclude any label not in the 3-class set (safety guard)
    merged = merged[merged["tier_a"].isin(CLASSES)].reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c not in (ACC_COL, "tier_a")]
    X = merged[feature_cols].values.astype(np.float32)

    # Encode labels to integers (0, 1, 2) in CLASSES order
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(merged["tier_a"].values)

    n_samples = len(X)
    class_dist = {CLASSES[i]: int((y == i).sum()) for i in range(3)}
    print(f"Training samples : {n_samples}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {class_dist}")

    # 2. Stratified 5-fold CV with per-fold bootstrap CI
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_records: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_val)
        f1_pt, f1_lo, f1_hi = bootstrap_macro_f1(y_val, y_pred)

        fold_records.append({
            "fold":     fold_idx,
            "n_train":  len(train_idx),
            "n_val":    len(val_idx),
            "macro_f1": f1_pt,
            "f1_lo":    f1_lo,   # 2.5th pct (95% CI lower bound)
            "f1_hi":    f1_hi,   # 97.5th pct
        })
        print(
            f"  Fold {fold_idx}: macro-F1 = {f1_pt:.4f}  "
            f"[{f1_lo:.4f}, {f1_hi:.4f}]"
        )

    cv_df = pd.DataFrame(fold_records)
    mean_f1 = float(cv_df["macro_f1"].mean())
    mean_lo = float(cv_df["f1_lo"].mean())
    mean_hi = float(cv_df["f1_hi"].mean())

    print(f"\n5-fold CV summary:")
    print(f"  mean macro-F1 : {mean_f1:.4f}")
    print(f"  mean CI lower : {mean_lo:.4f}  (2.5th pct, 95% CI)")
    print(f"  mean CI upper : {mean_hi:.4f}")

    # 3. Gate 2 evaluation
    gate2_pass = (mean_f1 >= 0.80) and (mean_lo >= 0.70)
    print(
        f"\nGate 2 (mean_f1 >= 0.80 AND mean_lo >= 0.70): "
        f"{'PASS ?' if gate2_pass else 'FAIL ?'}"
    )
    if not gate2_pass:
        print("  WARNING: Gate 2 not met -- review class balance and feature coverage.")

    # 4. Retrain on full dataset
    print("\nRetraining on full dataset...")
    final_clf = lgb.LGBMClassifier(**LGB_PARAMS)
    final_clf.fit(X, y)

    # 5. Save model
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"model": final_clf, "label_encoder": le, "feature_cols": feature_cols}, fh)
    print(f"Model saved -> {model_path}")

    # 6. Feature importance
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": final_clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_path = MODEL_DIR / "feature_importance.parquet"
    fi_df.to_parquet(fi_path, compression="zstd", index=False)
    print(f"Feature importance -> {fi_path}")
    print("\nTop-10 features:")
    print(fi_df.head(10).to_string(index=False))

    # 7. Save CV metrics
    cv_path = MODEL_DIR / "tier_a_cv_metrics.parquet"
    cv_df.to_parquet(cv_path, compression="zstd", index=False)

    summary = {
        "mean_macro_f1":   mean_f1,
        "mean_f1_lo_2.5":  mean_lo,
        "mean_f1_hi_97.5": mean_hi,
        "gate2_pass":      gate2_pass,
        "n_train":         n_samples,
        "class_distribution": class_dist,
        "n_features":      X.shape[1],
        "n_folds":         N_FOLDS,
        "n_bootstrap":     N_BOOT,
        "fold_results": fold_records,
    }
    (MODEL_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nCV metrics   -> {cv_path}")
    print(f"CV summary   -> {MODEL_DIR / 'cv_summary.json'}")


if __name__ == "__main__":
    run()
