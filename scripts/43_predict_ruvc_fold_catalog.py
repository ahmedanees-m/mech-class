"""Prospective prediction: RuvC-fold superfamily triage.

Runs mech-class over all atlas proteins NOT in the labeled training set that
carry at least one whitelist Pfam domain. Produces a triage table distinguishing:
  - True DSB_NUCLEASE (e.g. CRISPR-Cas nucleases)
  - DSB_FREE_TRANSEST_RECOMBINASE with IS110 composite flag (the key correction)
  - TRANSPOSASE / Other

This is the main prospective output demonstrating IS110 reclassification at
genome scale: proteins that InterPro CL0219 (RNase H-like clan) labels as
'nuclease' because of IS110's RuvC-like N-terminal fold are correctly reassigned
to DSB_FREE_TRANSEST_RECOMBINASE by the composite head.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install lightgbm --quiet && python scripts/43_predict_ruvc_fold_catalog.py"

Inputs:
  /data/graphs/atlas.duckdb                   -- protein metadata + domain edges
  /data/features/fused/feature_matrix.parquet -- training features (for exclusion)

Expected output:
  /data/results/ruvc_fold_triage.parquet
  /data/results/ruvc_fold_triage.tsv
  /data/results/ruvc_fold_triage_summary.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ATLAS_DB    = Path("/data/graphs/atlas.duckdb")
FEAT_MATRIX = Path("/data/features/fused/feature_matrix.parquet")
MODEL_DIR   = Path("/data/models")
TIER_A_PATH = MODEL_DIR / "tier_a" / "model.pkl"
COMP_PATH   = MODEL_DIR / "composite_head" / "model.pkl"
TIER_B_DIR  = MODEL_DIR / "tier_b"
OUT_DIR     = Path("/data/results")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pfam whitelist ? same order as dom_0..dom_22 in feature matrix
PFAM_WHITELIST = [
    "PF13395", "PF18541", "PF16595", "PF18516", "PF01548", "PF02371",
    "PF07282", "PF00665", "PF01609", "PF13586", "PF08721", "PF11426",
    "PF05621", "PF00589", "PF00239", "PF07508", "PF01844", "PF02486",
    "PF18061", "PF16592", "PF16593", "PF13639", "PF03377",
]
PFAM_TO_IDX = {p: i for i, p in enumerate(PFAM_WHITELIST)}

PRINT_EVERY = 500


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _load_models():
    with open(TIER_A_PATH, "rb") as f:
        ta = pickle.load(f)
    lgbm_a    = ta["model"]
    feat_cols = ta["feature_cols"]
    le_a      = ta["label_encoder"]

    with open(COMP_PATH, "rb") as f:
        comp = pickle.load(f)
    lgbm_comp      = comp["model"]
    comp_feat_cols = comp["feature_cols"]

    tier_b_models: dict[str, tuple] = {}
    if TIER_B_DIR.exists():
        for pkl in TIER_B_DIR.glob("*/model.pkl"):
            class_name = pkl.parent.name
            with open(pkl, "rb") as f:
                tb = pickle.load(f)
            tier_b_models[class_name] = (tb["model"], tb["feature_cols"], tb["label_encoder"])

    _log(f"Tier-A feature cols: {len(feat_cols)}")
    _log(f"Tier-B classes: {list(tier_b_models.keys())}")
    return lgbm_a, feat_cols, le_a, lgbm_comp, comp_feat_cols, tier_b_models


def _load_candidates(training_accs: set[str]) -> pd.DataFrame:
    """
    Load all atlas proteins that have at least one whitelist Pfam domain,
    excluding training proteins.

    Returns DataFrame: accession, organism, pfam_hits (pipe-separated string)
    """
    _log("Querying atlas for non-training proteins with whitelist Pfam domains...")
    con = duckdb.connect(str(ATLAS_DB), read_only=True)

    # Get all (protein_accession, pfam_accession) pairs for whitelist domains
    placeholders = ", ".join(["?" for _ in PFAM_WHITELIST])
    query = f"""
        SELECT
            p.accession,
            o.scientific_name AS organism,
            d.accession AS pfam_acc
        FROM nodes_protein p
        JOIN nodes_organism o ON p.organism_id = o.id
        JOIN edges e ON e.source_id = p.id AND e.source_type = 'Protein'
        JOIN nodes_domain d ON d.id = e.target_id AND e.target_type = 'Domain'
        WHERE d.accession IN ({placeholders})
    """
    rows = con.execute(query, PFAM_WHITELIST).fetchall()
    con.close()

    if not rows:
        _log("[WARN] No whitelist-domain proteins found in atlas.")
        return pd.DataFrame(columns=["accession", "organism", "pfam_hits"])

    # Aggregate pfam hits per protein
    from collections import defaultdict
    hits: dict[str, list[str]] = defaultdict(list)
    orgs: dict[str, str] = {}
    for acc, org, pfam in rows:
        hits[acc].append(pfam)
        orgs[acc] = org

    records = [
        {"accession": acc, "organism": orgs[acc], "pfam_hits": "|".join(sorted(set(hits[acc])))}
        for acc in hits
    ]
    df = pd.DataFrame(records)

    _log(f"  Atlas proteins with whitelist domain: {len(df)}")
    df = df[~df["accession"].isin(training_accs)].reset_index(drop=True)
    _log(f"  After training exclusion: {len(df)}")
    return df


def _build_feature_row(pfam_hits: list[str], feat_cols: list[str]) -> np.ndarray:
    """
    Domain-only feature row (seq and struct channels zero-filled).

    For the IS110 triage, domain features carry the primary signal.
    Proteins missing whitelist domains get all-zero vectors and typically
    receive the majority-class prediction (DSB_FREE) with low novelty value.
    """
    row = np.zeros(len(feat_cols), dtype=np.float32)
    col_map = {c: i for i, c in enumerate(feat_cols)}

    pfam_set = set(pfam_hits)
    wl_hits = []
    for wl_idx, pfam in enumerate(PFAM_WHITELIST):
        col = f"dom_{wl_idx}"
        if col in col_map and pfam in pfam_set:
            row[col_map[col]] = 1.0
            wl_hits.append(pfam)

    # IS110 composite flag
    if "dom_23" in col_map:
        row[col_map["dom_23"]] = float("PF01548" in pfam_set and "PF02371" in pfam_set)

    # Single-domain flag
    if "dom_25" in col_map:
        row[col_map["dom_25"]] = float(len(wl_hits) == 1)

    return row


def run() -> None:
    # ---- Training exclusion set ----
    training_accs: set[str] = set()
    if FEAT_MATRIX.exists():
        fm = pd.read_parquet(FEAT_MATRIX, columns=["uniprot_acc"])
        training_accs = set(fm["uniprot_acc"].tolist())
        _log(f"Training proteins (excluded from triage): {len(training_accs)}")

    # ---- Load models ----
    lgbm_a, feat_cols, le_a, lgbm_comp, comp_feat_cols, tier_b_models = _load_models()

    # ---- Load candidates ----
    candidates = _load_candidates(training_accs)
    n_total = len(candidates)
    if n_total == 0:
        _log("No candidates to process.")
        return

    # ---- Batch predict (domain features only ? fast) ----
    _log(f"Building feature matrix for {n_total} candidates...")

    # Pre-build all feature rows at once
    X = np.vstack([
        _build_feature_row(row["pfam_hits"].split("|"), feat_cols)
        for _, row in candidates.iterrows()
    ])
    X_df = pd.DataFrame(X, columns=feat_cols)

    _log("Running Tier-A predictions...")
    proba_a = lgbm_a.predict_proba(X_df)
    pred_idx = np.argmax(proba_a, axis=1)
    tier_a_labels = le_a.inverse_transform(pred_idx)
    tier_a_conf = proba_a[np.arange(len(proba_a)), pred_idx]

    _log("Running composite head predictions...")
    X_comp = X_df[comp_feat_cols] if comp_feat_cols else X_df
    comp_proba = lgbm_comp.predict_proba(X_comp)
    composite_bool = comp_proba[:, 1] >= 0.5
    composite_prob = comp_proba[:, 1]

    _log("Running Tier-B predictions...")
    tier_b_labels = np.full(n_total, None, dtype=object)
    for class_name, (lgbm_b, b_feat_cols, le_b) in tier_b_models.items():
        mask = tier_a_labels == class_name
        if mask.sum() == 0:
            continue
        X_b = X_df[b_feat_cols] if b_feat_cols else X_df
        proba_b = lgbm_b.predict_proba(X_b[mask])
        b_idx = np.argmax(proba_b, axis=1)
        tier_b_labels[mask] = le_b.inverse_transform(b_idx)

    # ---- Assemble records ----
    _log("Assembling results...")
    records = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        pfam_set = set(row["pfam_hits"].split("|"))
        is110 = "PF01548" in pfam_set and "PF02371" in pfam_set
        is110_reclassified = bool(is110 and composite_bool[i] and tier_a_labels[i] == "DSB_FREE_TRANSEST_RECOMBINASE")

        records.append({
            "accession":           row["accession"],
            "organism":            row["organism"],
            "pfam_hits":           row["pfam_hits"],
            "tier_a":              tier_a_labels[i],
            "tier_a_confidence":   float(tier_a_conf[i]),
            "tier_b":              tier_b_labels[i],
            "composite":           bool(composite_bool[i]),
            "composite_prob":      float(composite_prob[i]),
            "is110_reclassified":  is110_reclassified,
        })

        if (i + 1) % PRINT_EVERY == 0:
            _log(f"  {i+1}/{n_total} assembled...")

    df = pd.DataFrame(records)

    # Sort: IS110 reclassified first, then by tier-A confidence
    df_is110 = df[df["is110_reclassified"]].sort_values("tier_a_confidence", ascending=False)
    df_other = df[~df["is110_reclassified"]].sort_values("tier_a_confidence", ascending=False)
    df_out = pd.concat([df_is110, df_other], ignore_index=True)

    df_out.to_parquet(OUT_DIR / "ruvc_fold_triage.parquet", compression="zstd")
    df_out.to_csv(OUT_DIR / "ruvc_fold_triage.tsv", sep="\t", index=False)

    n_is110 = int(df_out["is110_reclassified"].sum())
    n_composite = int(df_out["composite"].sum())

    summary = {
        "total_candidates":              n_total,
        "is110_reclassified_count":      n_is110,
        "is110_reclassified_fraction":   float(n_is110 / max(n_total, 1)),
        "composite_flagged":             n_composite,
        "tier_a_distribution":           df_out["tier_a"].value_counts().to_dict(),
        "tier_b_distribution":           df_out["tier_b"].value_counts().dropna().to_dict(),
        "feature_channels":              "domain_only (seq/struct zero-filled)",
    }
    (OUT_DIR / "ruvc_fold_triage_summary.json").write_text(json.dumps(summary, indent=2))

    _log("")
    _log("=== RuvC-fold triage complete ===")
    _log(f"Total candidates         : {n_total}")
    _log(f"IS110 reclassified       : {n_is110}  ({100*n_is110/max(n_total,1):.1f}%)")
    _log(f"Composite flagged        : {n_composite}")
    _log(f"Tier-A distribution:")
    for cls, cnt in df_out["tier_a"].value_counts().items():
        _log(f"  {cls}: {cnt}")
    _log(f"Triage -> {OUT_DIR / 'ruvc_fold_triage.parquet'}")
    _log(f"TSV    -> {OUT_DIR / 'ruvc_fold_triage.tsv'}")


if __name__ == "__main__":
    run()
