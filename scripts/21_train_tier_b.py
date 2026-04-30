"""Train Tier-B sub-classifiers.

Most proteins are UNKNOWN at Tier-B; classifiers trained only where
>=2 sub-classes each have >=3 samples. Skips gracefully otherwise.

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
                 python scripts/21_train_tier_b.py"

Expected output:
  /data/models/tier_b/{tier_a_class}/model.pkl
  /data/models/tier_b/{tier_a_class}/cv_results.json
  /data/models/tier_b/tier_b_summary.json
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

FEAT_PATH  = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH = Path("/data/features/fused/labels.parquet")
TIER_B_DIR = Path("/data/models/tier_b")
TIER_B_DIR.mkdir(parents=True, exist_ok=True)

ACC_COL       = "uniprot_acc"
MIN_PER_CLASS = 3
MIN_CLASSES   = 2
SEED          = 42
N_BOOT        = 500

LGB_PARAMS_BASE = {
    "objective": "multiclass",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 3,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def _bootstrap_macro_f1(y_true, y_pred, n=N_BOOT, seed=SEED):
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
    merged = feat_df.merge(label_df[[ACC_COL, "tier_a", "tier_b"]],
                           on=ACC_COL, how="inner")
    merged = merged.dropna(subset=["tier_a", "tier_b"])
    feature_cols = [c for c in merged.columns
                    if c not in (ACC_COL, "tier_a", "tier_b")]

    tier_a_classes = sorted(merged["tier_a"].unique().tolist())
    print("Tier-A classes:", tier_a_classes)
    print("Tier-B distribution:")
    print(merged.groupby(["tier_a", "tier_b"]).size().to_string())
    print()

    summary = {}

    for tier_a_cls in tier_a_classes:
        subset = merged[merged["tier_a"] == tier_a_cls].copy()
        counts = subset["tier_b"].value_counts()
        valid_cls = [c for c, n in counts.items() if n >= MIN_PER_CLASS]

        if len(valid_cls) < MIN_CLASSES:
            reason = (
                "Only " + str(len(valid_cls)) + " Tier-B classes with >=" +
                str(MIN_PER_CLASS) + " samples (need " + str(MIN_CLASSES) + "). "
                "Distribution: " + str(dict(counts))
            )
            print("[SKIP] " + tier_a_cls + ": " + reason)
            summary[tier_a_cls] = {
                "skipped": True, "reason": reason,
                "tier_b_distribution": {str(k): int(v) for k, v in counts.items()},
            }
            continue

        mask = subset["tier_b"].isin(valid_cls)
        sub  = subset[mask].copy()
        X    = sub[feature_cols].values.astype(np.float32)
        y_raw = sub["tier_b"].values
        le   = LabelEncoder()
        y    = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        params = {**LGB_PARAMS_BASE, "num_class": n_classes}

        n_folds = min(5, int(min(counts[c] for c in valid_cls)))
        print("Training: " + tier_a_cls +
              " | classes=" + str(list(le.classes_)) +
              " | n=" + str(len(X)) + " | folds=" + str(n_folds))

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold_f1s, fold_los = [], []
        for tr, va in skf.split(X, y):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X[tr], y[tr])
            y_pred = clf.predict(X[va])
            f1_pt, f1_lo, _ = _bootstrap_macro_f1(y[va], y_pred)
            fold_f1s.append(f1_pt)
            fold_los.append(f1_lo)

        mean_f1 = float(np.mean(fold_f1s))
        mean_lo = float(np.mean(fold_los))
        print("  CV mean macro-F1: %.4f  CI_lo: %.4f" % (mean_f1, mean_lo))

        final_clf = lgb.LGBMClassifier(**params)
        final_clf.fit(X, y)

        fanzor_result = None
        if "N2_Fanzor_OMEGA" in list(le.classes_):
            fi = list(le.classes_).index("N2_Fanzor_OMEGA")
            fm = (y == fi)
            if fm.sum() > 0:
                fc_val = float(final_clf.predict_proba(X)[fm, fi].mean())
                fanzor_result = {
                    "mean_confidence": fc_val,
                    "criterion_met": fc_val >= 0.70,
                }
                print("  Fanzor confidence: %.4f (%s)" % (
                    fc_val, "PASS" if fc_val >= 0.70 else "FAIL"))

        out_dir = TIER_B_DIR / tier_a_cls.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump({
                "model": final_clf,
                "label_encoder": le,
                "feature_cols": feature_cols,
            }, fh)

        results = {
            "tier_a_class": tier_a_cls,
            "tier_b_classes": list(le.classes_),
            "n_train": int(len(X)),
            "n_folds": n_folds,
            "mean_macro_f1": mean_f1,
            "mean_f1_lo": mean_lo,
            "fold_f1s": fold_f1s,
            "fold_los": fold_los,
            "fanzor": fanzor_result,
            "skipped": False,
        }
        (out_dir / "cv_results.json").write_text(json.dumps(results, indent=2))
        summary[tier_a_cls] = results
        print("  Saved -> " + str(model_path))

    (TIER_B_DIR / "tier_b_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=== Tier-B complete ===")
    print("Summary -> " + str(TIER_B_DIR / "tier_b_summary.json"))


if __name__ == "__main__":
    run()
