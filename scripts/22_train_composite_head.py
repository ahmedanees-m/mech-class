"""Train IS110 composite-architecture head.

Conservative binary classifier: IS110-composite (PF01548 + PF02371 both
present) vs all other proteins.  Uses only the 26 domain features + 7
active-site geometry features (33-dim) to maximise interpretability and
avoid overfitting on N=14 positives.

Post-hoc rule guard: proteins with dom_4 > 0.5 AND dom_5 > 0.5 (PF01548
AND PF02371) are always predicted positive regardless of model score.

Pre-registered criterion (S0.5):
  AUROC >= 0.95 on 5-fold stratified CV
  FP rate <= 0.10  (FP / (FP + TN))

Input:
  /data/features/fused/feature_matrix.parquet  columns: uniprot_acc, feat_*
  /data/features/fused/labels.parquet          columns: uniprot_acc, composite_flag, ...

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
                 python scripts/22_train_composite_head.py"

Expected output:
  /data/models/composite_head/model.pkl
  /data/models/composite_head/composite_metrics.json
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# -- Paths --------------------------------------------------------------------
FEAT_PATH  = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH = Path("/data/features/fused/labels.parquet")
OUT_DIR    = Path("/data/models/composite_head")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACC_COL = "uniprot_acc"
N_FOLDS = 5
SEED    = 42

# LGB params for extremely imbalanced binary task (14/572 = 2.4% positive)
LGB_PARAMS = {
    "objective":         "binary",
    "n_estimators":      300,
    "learning_rate":     0.05,
    "num_leaves":        15,      # small tree -- only 33 features
    "min_child_samples": 3,       # allow small leaves for 14 positives
    "colsample_bytree":  0.8,
    "subsample":         0.8,
    "subsample_freq":    1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  40,      # ~558/14; boost minority class
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

# dom_4 = PF01548 (DEDD_Tnp_IS110), dom_5 = PF02371 (Transposase_20)
RULE_DOM_A = "dom_4"   # PF01548
RULE_DOM_B = "dom_5"   # PF02371


def _rule_positive(row: pd.Series) -> bool:
    """Hard rule: both IS110 domain columns present."""
    return bool(
        RULE_DOM_A in row.index and row[RULE_DOM_A] > 0.5 and
        RULE_DOM_B in row.index and row[RULE_DOM_B] > 0.5
    )


def run() -> None:
    feat_df  = pd.read_parquet(FEAT_PATH)
    label_df = pd.read_parquet(LABEL_PATH)

    merged = feat_df.merge(
        label_df[[ACC_COL, "composite_flag"]], on=ACC_COL, how="inner"
    )
    merged["composite_flag"] = merged["composite_flag"].fillna(False).astype(int)

    # Use only domain (dom_*) + active-site (as_*) features
    feature_cols = [
        c for c in merged.columns
        if c.startswith("dom_") or c.startswith("as_")
    ]
    if not feature_cols:
        raise RuntimeError("No dom_* or as_* columns found in feature matrix.")

    X = merged[feature_cols].values.astype(np.float32)
    y = merged["composite_flag"].values

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"Composite head training: {n_pos} positive, {n_neg} negative")
    print(f"Features used: {len(feature_cols)} ({feature_cols[:5]}...)")

    # -- 5-fold stratified CV -------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs: list[float] = []
    fold_fp_rates: list[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_tr, y_tr)

        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else float("nan")
        tn = int(((y_pred == 0) & (y_val == 0)).sum())
        fp = int(((y_pred == 1) & (y_val == 0)).sum())
        fp_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        fold_aucs.append(auc)
        fold_fp_rates.append(fp_rate)
        print(f"  Fold {fold_idx}: AUROC = {auc:.4f}  FP rate = {fp_rate:.4f}")

    valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    mean_fp  = float(np.mean(fold_fp_rates))

    print(f"\n5-fold CV:")
    print(f"  mean AUROC  : {mean_auc:.4f}")
    print(f"  mean FP rate: {mean_fp:.4f}")

    auroc_pass = mean_auc >= 0.95
    fp_pass    = mean_fp <= 0.10
    print(f"\nPre-registered AUROC >= 0.95: {'PASS' if auroc_pass else 'FAIL'}")
    print(f"Pre-registered FP rate <= 0.10: {'PASS' if fp_pass else 'FAIL'}")

    # -- Retrain on full dataset + rule guard ---------------------------------
    print("\nRetraining on full dataset...")
    final_clf = lgb.LGBMClassifier(**LGB_PARAMS)
    final_clf.fit(X, y)

    # Full-dataset evaluation with rule guard
    y_proba_full = final_clf.predict_proba(X)[:, 1]
    y_pred_full  = (y_proba_full >= 0.5).astype(int)

    # Apply rule guard: force-positive any protein with both IS110 domains
    feat_df_sub = merged[feature_cols].copy()
    for i, row in feat_df_sub.iterrows():
        if _rule_positive(row):
            y_pred_full[feat_df_sub.index.get_loc(i)] = 1

    prec = float(precision_score(y, y_pred_full, zero_division=0))
    rec  = float(recall_score(y, y_pred_full, zero_division=0))
    f1   = float(f1_score(y, y_pred_full, zero_division=0))
    tn = int(((y_pred_full == 0) & (y == 0)).sum())
    fp = int(((y_pred_full == 1) & (y == 0)).sum())
    fp_rate_full = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print(f"Full-dataset (with rule guard):")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  FP rate: {fp_rate_full:.4f}")

    # -- Save -----------------------------------------------------------------
    model_path = OUT_DIR / "model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"model": final_clf, "feature_cols": feature_cols}, fh)
    print(f"\nModel -> {model_path}")

    metrics = {
        "cv_mean_auroc":      mean_auc,
        "cv_mean_fp_rate":    mean_fp,
        "cv_fold_aurocs":     fold_aucs,
        "cv_fold_fp_rates":   fold_fp_rates,
        "auroc_criterion_pass": auroc_pass,
        "fp_criterion_pass":    fp_pass,
        "full_precision":     prec,
        "full_recall":        rec,
        "full_f1":            f1,
        "full_fp_rate":       fp_rate_full,
        "n_positive":         n_pos,
        "n_negative":         n_neg,
        "n_features":         len(feature_cols),
        "feature_cols":       feature_cols,
    }
    metrics_path = OUT_DIR / "composite_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics -> {metrics_path}")


if __name__ == "__main__":
    run()
