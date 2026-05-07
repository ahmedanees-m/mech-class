"""Step 15 — Channel ablation study (Week 6).

Trains Tier-A classifier with each channel removed (leave-one-out ablation)
to quantify per-channel contribution to macro-F1.

Ablation conditions:
  full        : F_seq + F_struct + F_domain + F_active_site  (baseline)
  -seq        : F_struct + F_domain + F_active_site
  -struct     : F_seq + F_domain + F_active_site
  -domain     : F_seq + F_struct + F_active_site
  -active_site: F_seq + F_struct + F_domain
  seq_only    : F_seq only
  domain_only : F_domain only

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/23_channel_ablation.py"

Expected output:
  /data/models/ablation/ablation_results.json
  /data/models/ablation/ablation_results.parquet
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mech_class.models.lightgbm_clf import MechLGBMClassifier

X_PATH = Path("/data/features/X.parquet")
Y_PATH = Path("/data/features/y.parquet")
OUT_DIR = Path("/data/models/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_channel_cols(feature_cols: list[str]) -> dict[str, list[str]]:
    return {
        "seq": [c for c in feature_cols if c.startswith("seq_")],
        "struct": [c for c in feature_cols if c.startswith("struct_")],
        "domain": [c for c in feature_cols if c.startswith("dom_") or c == "is110_composite"],
        "active_site": [c for c in feature_cols if c.startswith("as_")],
    }


def _run_condition(
    label: str,
    cols: list[str],
    X_df: pd.DataFrame,
    y: np.ndarray,
) -> dict:
    if not cols:
        return {"label": label, "n_features": 0, "macro_f1_mean": None, "macro_f1_std": None,
                "bootstrap_f1": None, "ci_lower": None, "ci_upper": None, "skipped": True}

    X = X_df[cols].values.astype(np.float32)
    clf = MechLGBMClassifier(n_classes=3, random_seed=42)
    clf.fit(X, y, feature_names=cols)
    cv = clf.cross_validate(X, y, n_folds=5)
    f1_pt, ci_lo, ci_hi = clf.macro_f1(X, y, n_bootstrap=1000, seed=42)
    return {
        "label": label,
        "n_features": len(cols),
        "macro_f1_mean": cv["macro_f1_mean"],
        "macro_f1_std": cv["macro_f1_std"],
        "bootstrap_f1": f1_pt,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "skipped": False,
    }


def run() -> None:
    X_df = pd.read_parquet(X_PATH)
    y_df = pd.read_parquet(Y_PATH)

    merged = X_df.merge(y_df[["accession", "tier_a"]], on="accession", how="inner")
    merged = merged.dropna(subset=["tier_a"])

    feature_cols = [c for c in merged.columns if c not in ("accession", "tier_a", "tier_b", "composite_flag")]
    y = merged["tier_a"].values
    X_df_sub = merged[feature_cols]

    channels = _get_channel_cols(feature_cols)
    all_cols = feature_cols

    conditions = {
        "full": all_cols,
        "-seq": [c for c in all_cols if c not in channels["seq"]],
        "-struct": [c for c in all_cols if c not in channels["struct"]],
        "-domain": [c for c in all_cols if c not in channels["domain"]],
        "-active_site": [c for c in all_cols if c not in channels["active_site"]],
        "seq_only": channels["seq"],
        "domain_only": channels["domain"],
    }

    results = []
    for label, cols in conditions.items():
        print(f"Running ablation: {label} ({len(cols)} features)...")
        res = _run_condition(label, cols, X_df_sub, y)
        results.append(res)
        if not res["skipped"]:
            print(f"  F1: {res['bootstrap_f1']:.3f} [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")

    df = pd.DataFrame(results)
    df.to_parquet(OUT_DIR / "ablation_results.parquet", compression="zstd")

    full_f1 = next(r["bootstrap_f1"] for r in results if r["label"] == "full")
    for r in results:
        if r["bootstrap_f1"] is not None:
            r["delta_vs_full"] = r["bootstrap_f1"] - full_f1

    (OUT_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Channel ablation complete ===")
    print(f"{'Condition':<20} {'F1':>6}  {'CI':>14}  {'ΔFULL':>7}")
    print("-" * 55)
    for r in results:
        if r["skipped"]:
            continue
        delta = f"{r.get('delta_vs_full', 0):+.3f}"
        ci = f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]"
        print(f"{r['label']:<20} {r['bootstrap_f1']:>6.3f}  {ci:>14}  {delta:>7}")

    print(f"\nResults → {OUT_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    run()
