from __future__ import annotations
import json, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FEAT_PATH  = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH = Path("/data/features/fused/labels.parquet")
OUT_DIR    = Path("/data/models/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACC_COL = "uniprot_acc"
N_FOLDS = 5
N_BOOT  = 1000
SEED    = 42
CLASSES = ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE", "TRANSPOSASE"]

LGB_PARAMS = {
    "objective": "multiclass", "num_class": 3, "n_estimators": 400,
    "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 5,
    "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1,
    "reg_lambda": 0.1, "class_weight": "balanced",
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
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

def _run_condition(label, feat_cols, X_df, y):
    if not feat_cols:
        print("  [" + label + "] SKIPPED -- no features")
        return {"condition": label, "n_features": 0, "skipped": True}
    t0 = time.time()
    X = X_df[feat_cols].values.astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_f1s, fold_los, fold_his = [], [], []
    for tr, va in skf.split(X, y):
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[va])
        f1_pt, f1_lo, f1_hi = _bootstrap_macro_f1(y[va], y_pred)
        fold_f1s.append(f1_pt); fold_los.append(f1_lo); fold_his.append(f1_hi)
    mean_f1 = float(np.mean(fold_f1s))
    mean_lo = float(np.mean(fold_los))
    mean_hi = float(np.mean(fold_his))
    elapsed = time.time() - t0
    print("  [" + label + "]  n_feat=" + str(len(feat_cols)) +
          "  mean_F1=" + "%.4f" % mean_f1 + "  CI_lo=" + "%.4f" % mean_lo +
          "  (" + "%.1f" % elapsed + "s)")
    return {"condition": label, "n_features": len(feat_cols),
            "mean_f1": mean_f1, "mean_f1_lo": mean_lo, "mean_f1_hi": mean_hi,
            "fold_f1s": fold_f1s, "fold_los": fold_los, "skipped": False}

def run():
    feat_df  = pd.read_parquet(FEAT_PATH)
    label_df = pd.read_parquet(LABEL_PATH)
    merged = feat_df.merge(label_df[[ACC_COL, "tier_a"]], on=ACC_COL, how="inner")
    merged = merged.dropna(subset=["tier_a"])
    merged = merged[merged["tier_a"].isin(CLASSES)].reset_index(drop=True)
    feature_cols = [c for c in merged.columns if c not in (ACC_COL, "tier_a")]
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(merged["tier_a"].values)
    X_df = merged[feature_cols]
    seq_cols    = [c for c in feature_cols if c.startswith("seq_")]
    struct_cols = [c for c in feature_cols if c.startswith("struct_")]
    dom_cols    = [c for c in feature_cols if c.startswith("dom_")]
    as_cols     = [c for c in feature_cols if c.startswith("as_")]
    all_cols    = seq_cols + struct_cols + dom_cols + as_cols
    print("Samples: " + str(len(y)) + " | seq=" + str(len(seq_cols)) +
          " struct=" + str(len(struct_cols)) +
          " dom=" + str(len(dom_cols)) + " as=" + str(len(as_cols)))
    conditions = {
        "full":         all_cols,
        "-seq":         struct_cols + dom_cols + as_cols,
        "-struct":      seq_cols    + dom_cols + as_cols,
        "-domain":      seq_cols    + struct_cols + as_cols,
        "-active_site": seq_cols    + struct_cols + dom_cols,
        "seq_only":     seq_cols,
        "struct_only":  struct_cols,
        "domain_only":  dom_cols,
    }
    results = []
    for label, cols in conditions.items():
        res = _run_condition(label, cols, X_df, y)
        results.append(res)
    full_f1 = next(r.get("mean_f1", float("nan")) for r in results
                   if r["condition"] == "full")
    for r in results:
        if not r.get("skipped") and "mean_f1" in r:
            r["delta_vs_full"] = round(r["mean_f1"] - full_f1, 4)
    rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in results]
    pd.DataFrame(rows).to_parquet(OUT_DIR / "ablation_results.parquet",
                                  compression="zstd", index=False)
    (OUT_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print()
    print("Condition        n_feat  mean_F1   CI_lo   delta")
    print("-" * 52)
    for r in results:
        if r.get("skipped"):
            print(r["condition"] + " SKIPPED")
            continue
        delta = "%+.4f" % r.get("delta_vs_full", 0.0)
        print("%-16s %7d %8.4f %7.4f %7s" % (
            r["condition"], r["n_features"], r["mean_f1"], r["mean_f1_lo"], delta))
    print("Results -> " + str(OUT_DIR / "ablation_results.parquet"))

if __name__ == "__main__":
    run()
