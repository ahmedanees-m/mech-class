"""Step 18 -- Bootstrap CI aggregation for manuscript Table 1 (Week 6).

Loads saved Tier-A and composite-head models, recomputes macro-F1 and
per-class F1 with 1000x bootstrap CIs (2.5th/97.5th pct, 95% CI).

Input:
  /data/features/fused/feature_matrix.parquet
  /data/features/fused/labels.parquet
  /data/models/tier_a/model.pkl
  /data/models/composite_head/model.pkl

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
                 python scripts/24_bootstrap_cis.py"

Expected output: /data/models/bootstrap_ci_summary.json
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

FEAT_PATH       = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH      = Path("/data/features/fused/labels.parquet")
TIER_A_MODEL    = Path("/data/models/tier_a/model.pkl")
COMPOSITE_MODEL = Path("/data/models/composite_head/model.pkl")
OUT             = Path("/data/models/bootstrap_ci_summary.json")

ACC_COL = "uniprot_acc"
N_BOOT  = 1000
SEED    = 42
CLASSES = ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE", "TRANSPOSASE"]


def _ci(scores, alpha=0.05):
    return (float(np.quantile(scores, alpha / 2)),
            float(np.quantile(scores, 1 - alpha / 2)))


def run():
    feat_df  = pd.read_parquet(FEAT_PATH)
    label_df = pd.read_parquet(LABEL_PATH)
    merged = feat_df.merge(
        label_df[[ACC_COL, "tier_a", "composite_flag"]], on=ACC_COL, how="inner"
    )
    merged = merged.dropna(subset=["tier_a"])
    merged = merged[merged["tier_a"].isin(CLASSES)].reset_index(drop=True)

    if not TIER_A_MODEL.exists():
        print("ERROR: " + str(TIER_A_MODEL) + " not found. Run 20_train_tier_a.py first.")
        return

    with open(TIER_A_MODEL, "rb") as fh:
        pkg = pickle.load(fh)
    clf_a     = pkg["model"]
    feat_cols = pkg["feature_cols"]
    le_a      = pkg["label_encoder"]

    X      = merged[feat_cols].values.astype(np.float32)
    y_true = le_a.transform(merged["tier_a"].values)
    y_pred = clf_a.predict(X)

    rng = np.random.default_rng(SEED)

    # Macro-F1
    macro_samples = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        macro_samples.append(float(f1_score(y_true[idx], y_pred[idx],
                                            average="macro", zero_division=0)))
    macro_pt = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_lo, macro_hi = _ci(np.array(macro_samples))
    print("Tier-A Macro-F1: %.4f [%.4f, %.4f]" % (macro_pt, macro_lo, macro_hi))

    # Per-class F1
    per_class = {}
    for cls in CLASSES:
        ci_idx = list(le_a.classes_).index(cls)
        ytb = (y_true == ci_idx).astype(int)
        ypb = (y_pred == ci_idx).astype(int)
        pt = float(f1_score(ytb, ypb, average="binary", zero_division=0))
        samps = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(ytb), size=len(ytb))
            samps.append(float(f1_score(ytb[idx], ypb[idx],
                                        average="binary", zero_division=0)))
        lo, hi = _ci(np.array(samps))
        per_class[cls] = {"f1_point": pt, "ci_lower": lo, "ci_upper": hi}
        print("  " + cls + ": F1=%.4f [%.4f, %.4f]" % (pt, lo, hi))

    # Composite head
    composite_result = None
    if COMPOSITE_MODEL.exists():
        with open(COMPOSITE_MODEL, "rb") as fh:
            pkg_c = pickle.load(fh)
        clf_c   = pkg_c["model"]
        fc_cols = pkg_c["feature_cols"]
        Xc  = merged[fc_cols].values.astype(np.float32)
        yc  = merged["composite_flag"].fillna(False).astype(int).values
        ypc = clf_c.predict(Xc)
        tn  = int(((ypc == 0) & (yc == 0)).sum())
        fp  = int(((ypc == 1) & (yc == 0)).sum())
        fp_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        try:
            ypc_proba = clf_c.predict_proba(Xc)[:, 1]
            auroc = float(roc_auc_score(yc, ypc_proba))
        except Exception:
            auroc = float("nan")
        fp_samps = []
        for _ in range(N_BOOT):
            idx  = rng.integers(0, len(yc), size=len(yc))
            yp_  = ypc[idx]
            yt_  = yc[idx]
            tn_  = int(((yp_ == 0) & (yt_ == 0)).sum())
            fp_  = int(((yp_ == 1) & (yt_ == 0)).sum())
            fp_samps.append(float(fp_ / (fp_ + tn_)) if (fp_ + tn_) > 0 else 0.0)
        fp_lo, fp_hi = _ci(np.array(fp_samps))
        composite_result = {
            "auroc": auroc,
            "fp_rate": fp_rate,
            "fp_rate_ci_lower": fp_lo,
            "fp_rate_ci_upper": fp_hi,
            "fp_criterion_pass": bool(fp_rate <= 0.10),
        }
        print("Composite AUROC: %.4f  FP rate: %.4f [%.4f, %.4f]" % (
            auroc, fp_rate, fp_lo, fp_hi))
    else:
        print("WARNING: composite model not found at " + str(COMPOSITE_MODEL))

    results = {
        "tier_a_macro_f1": {
            "point": macro_pt,
            "ci_lower": macro_lo,
            "ci_upper": macro_hi,
            "criterion_met": bool(macro_pt >= 0.80 and macro_lo >= 0.70),
        },
        "per_class_f1": per_class,
        "composite_head": composite_result,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "n_proteins": int(len(merged)),
    }
    OUT.write_text(json.dumps(results, indent=2))
    crit = results["tier_a_macro_f1"]["criterion_met"]
    print()
    print("=== Bootstrap CI Summary ===")
    print("Criterion (>=0.80, lo>=0.70): " + ("PASS" if crit else "FAIL"))
    print("Output -> " + str(OUT))


if __name__ == "__main__":
    run()
