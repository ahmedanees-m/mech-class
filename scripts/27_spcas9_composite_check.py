"""Diagnostic: SpCas9 composite head raw probability check.

Resolves the table vs. analysis text contradiction for Q99ZW2 (SpCas9):
  - Table printed composite=False
  - Analysis text said composite=True (FP)

Loads holdout_features.parquet, finds composite model, re-runs predict_proba
on Q99ZW2 row, and reports the ground truth.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install lightgbm scikit-learn --quiet && \\
                 python scripts/27_spcas9_composite_check.py"

Output:
  stdout only (no file written)
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR       = Path("/data")
FEAT_PATH      = Path("/data/validation/holdout_features.parquet")
RESULTS_PATH   = Path("/data/validation/holdout_results.json")
MODEL_DIR      = Path("/data/models")

TIER_A_PATH    = MODEL_DIR / "lgbm_tier_a" / "model.pkl"
CAS9_ACC       = "Q99ZW2"

PFAM_WHITELIST = [
    "PF13395","PF18541","PF16595","PF18516","PF01548","PF02371","PF07282",
    "PF00665","PF01609","PF13586","PF08721","PF11426","PF05621","PF00589",
    "PF00239","PF07508","PF01844","PF02486","PF18061","PF16592","PF16593",
    "PF13639","PF03377",
]


def main():
    # 1. Print existing holdout_results.json
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print("=== Existing holdout_results.json ===")
        for r in results:
            print(f"  {r.get('name','?'):30s}  acc={r.get('accession','?')}  "
                  f"tier_a={r.get('tier_a_pred','?'):30s}  "
                  f"composite={r.get('composite','?')}")
        print()
    else:
        print("[WARN] holdout_results.json not found\n")

    # 2. Load holdout feature matrix
    if not FEAT_PATH.exists():
        print(f"[ERROR] {FEAT_PATH} not found - cannot run composite check")
        return
    df = pd.read_parquet(FEAT_PATH)
    print(f"holdout_features shape: {df.shape}")
    print(f"Columns sample: {df.columns.tolist()[:8]}")
    if "accession" in df.columns:
        print(f"Accessions: {df['accession'].tolist()}")
    print()

    # 3. Load Tier-A model to get canonical feature_cols
    with open(TIER_A_PATH, "rb") as fh:
        tier_a_bundle = pickle.load(fh)
    feat_cols = tier_a_bundle["feature_cols"]
    print(f"Tier-A feature_cols: {len(feat_cols)} features")

    # 4. Identify Q99ZW2 row
    if "accession" in df.columns:
        cas9_df = df[df["accession"] == CAS9_ACC].copy()
    else:
        # Probe order: IS110=0, Fanzor=1, SpCas9=2, Bxb1=3, Tn5=4
        cas9_df = df.iloc[[2]].copy()

    if len(cas9_df) == 0:
        print(f"[ERROR] Q99ZW2 not found in holdout_features.parquet")
        return
    print(f"SpCas9 row found ({len(cas9_df)} rows)")

    # Align to Tier-A feat_cols
    for col in feat_cols:
        if col not in cas9_df.columns:
            cas9_df[col] = 0.0
    X_cas9 = cas9_df[feat_cols].values.astype(np.float32)

    # 5. Search for composite model
    composite_candidates = (
        list(MODEL_DIR.rglob("composite*.pkl"))
        + list(MODEL_DIR.rglob("*composite*.pkl"))
        + list(MODEL_DIR.rglob("composite_head*"))
    )
    composite_candidates = list(dict.fromkeys(composite_candidates))  # dedupe
    print(f"\nComposite model candidates found: {composite_candidates}")

    if not composite_candidates:
        print("[WARN] No composite model file found. Checking model directory structure:")
        for p in sorted(MODEL_DIR.rglob("*.pkl")):
            print(f"  {p}")
        print()
        # Fallback: check composite from inside tier_a bundle itself
        print("Checking tier_a bundle keys:", list(tier_a_bundle.keys()))
        _check_via_tier_a_model(tier_a_bundle, X_cas9, feat_cols, cas9_df)
        return

    # 6. Load composite model and run predict_proba
    comp_path = composite_candidates[0]
    with open(comp_path, "rb") as fh:
        comp_bundle = pickle.load(fh)

    print(f"\nLoaded composite model: {comp_path}")
    if isinstance(comp_bundle, dict):
        print(f"Bundle keys: {list(comp_bundle.keys())}")
        comp_model   = comp_bundle.get("model", comp_bundle)
        comp_fcols   = comp_bundle.get("feature_cols", feat_cols)
    else:
        comp_model   = comp_bundle
        comp_fcols   = feat_cols

    # Align features
    for col in comp_fcols:
        if col not in cas9_df.columns:
            cas9_df[col] = 0.0
    X_comp = cas9_df[comp_fcols].values.astype(np.float32)

    _run_composite(comp_model, X_comp)


def _run_composite(model, X):
    """Run composite model and print verdict."""
    print(f"\n=== SpCas9 (Q99ZW2) Composite Head ===")
    try:
        proba = model.predict_proba(X)[0]
        pred  = model.predict(X)[0]
        print(f"  predict_proba : {proba}")
        print(f"  predict       : {pred}")
        print(f"  composite_prob: {proba[1]:.6f}" if len(proba) > 1 else f"  prob: {proba[0]:.6f}")
        flag = (proba[1] > 0.5) if len(proba) > 1 else bool(pred)
        if flag:
            print("  -> composite = TRUE  (genuine FALSE POSITIVE - model over-fires)")
            print("    Note: This means the analysis text was correct; table was wrong.")
            print("    FP rate on holdout = 1/4 DSB_NUCLEASE probes")
        else:
            print("  -> composite = FALSE  (correct; table was right; analysis text was wrong)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        try:
            pred = model.predict(X)[0]
            print(f"  predict only: {pred}")
            print(f"  composite = {bool(pred)}")
        except Exception as e2:
            print(f"  [ERROR predict] {e2}")


def _check_via_tier_a_model(bundle, X, feat_cols, cas9_df):
    """Fallback: maybe composite is embedded in Tier-A bundle."""
    if "composite_head" in bundle:
        print("\nFound composite_head in Tier-A bundle!")
        _run_composite(bundle["composite_head"], X)
    elif "composite" in bundle:
        print("\nFound 'composite' key in Tier-A bundle!")
        comp = bundle["composite"]
        if isinstance(comp, dict):
            m = comp.get("model", comp)
            fc = comp.get("feature_cols", feat_cols)
            for col in fc:
                if col not in cas9_df.columns:
                    cas9_df[col] = 0.0
            Xc = cas9_df[fc].values.astype(np.float32)
            _run_composite(m, Xc)
        else:
            _run_composite(comp, X)
    else:
        print(f"No composite head found in bundle. Keys: {list(bundle.keys())}")
        print("\nManual check: SpCas9 domain flags (dom_*):")
        dom_cols = [c for c in cas9_df.columns if c.startswith("dom_")]
        for i, col in enumerate(dom_cols):
            val = float(cas9_df[col].iloc[0])
            if val != 0:
                print(f"  {col} (idx {i} = {PFAM_WHITELIST[i] if i < len(PFAM_WHITELIST) else '?'}): {val}")
        print("\nComposite domain flags (dom_23=IS110, dom_24=editor_fusion, dom_25=single_domain):")
        for ci in [23, 24, 25]:
            cname = f"dom_{ci}"
            if cname in cas9_df.columns:
                print(f"  {cname}: {float(cas9_df[cname].iloc[0])}")


if __name__ == "__main__":
    main()
