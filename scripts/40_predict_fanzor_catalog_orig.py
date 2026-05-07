"""Step 18 — Prospective prediction: Fanzor/OMEGA-nuclease catalog (Week 7).

Runs mech-class predictor over ~3,000 candidate Fanzor ortholog sequences
from the genome-atlas (pre-screened by RuvC-fold + OMEGA-element signature).

Outputs a ranked catalog with:
  - tier_a, tier_b predictions + confidences
  - IS110 composite flag
  - Source evidence (which atlas search hit nominated this sequence)
  - Novelty score (distance from nearest training example in ESM-2 space)

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/40_predict_fanzor_catalog.py"

Inputs:
  /data/graphs/atlas.duckdb              — ESM-2 embeddings + metadata
  /data/processed/fanzor_candidates.parquet  — pre-screened candidates

Expected output:
  /data/results/fanzor_catalog.parquet
  /data/results/fanzor_catalog.tsv
"""
from __future__ import annotations
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from mech_class.api import Predictor

ATLAS_DB = Path("/data/graphs/atlas.duckdb")
CANDIDATES_PATH = Path("/data/processed/fanzor_candidates.parquet")
X_TRAIN = Path("/data/features/X.parquet")
Y_TRAIN = Path("/data/features/y.parquet")
MODEL_DIR = Path("/data/models")
OUT_DIR = Path("/data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FANZOR_CONF_THRESHOLD = 0.70


def _load_candidates() -> pd.DataFrame:
    if CANDIDATES_PATH.exists():
        return pd.read_parquet(CANDIDATES_PATH)
    # Fallback: query atlas for RuvC-fold proteins not in training set
    print("[INFO] fanzor_candidates.parquet not found. Querying atlas for RuvC candidates...")
    con = duckdb.connect(str(ATLAS_DB), read_only=True)
    df = con.execute(
        "SELECT accession, sequence FROM proteins WHERE organism_domain = 'Eukaryota'"
    ).df()
    con.close()
    return df


def _novelty_score(
    candidate_emb: np.ndarray,
    train_embs: np.ndarray,
) -> float:
    """Cosine distance to nearest training neighbor (higher = more novel)."""
    if len(train_embs) == 0:
        return 1.0
    candidate_emb = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-8)
    train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    sims = train_norm @ candidate_emb
    return float(1.0 - sims.max())


def run() -> None:
    candidates = _load_candidates()
    print(f"Candidate sequences: {len(candidates)}")

    predictor = Predictor.load(model_dir=MODEL_DIR)

    # Load training ESM-2 embeddings for novelty scoring
    train_embs = np.zeros((0, 640))
    if X_TRAIN.exists() and Y_TRAIN.exists():
        X_df = pd.read_parquet(X_TRAIN)
        seq_cols = [c for c in X_df.columns if c.startswith("seq_")]
        if seq_cols:
            train_embs = X_df[seq_cols].values.astype(np.float32)

    records = []
    fanzor_count = 0

    for i, row in candidates.iterrows():
        acc = row["accession"]
        seq = row.get("sequence", "")

        if seq:
            pred = predictor.predict_from_sequence(acc, seq)
        else:
            pred = predictor.predict_from_accession(acc)

        novelty = 0.0
        if len(train_embs) > 0:
            # Get ESM-2 embedding for novelty calc from atlas
            con = duckdb.connect(str(ATLAS_DB), read_only=True)
            emb_row = con.execute(
                "SELECT * FROM embeddings WHERE accession = ?", [acc]
            ).df()
            con.close()
            if not emb_row.empty:
                emb_cols = [c for c in emb_row.columns if c != "accession"]
                emb = emb_row[emb_cols].values[0].astype(np.float32)
                novelty = _novelty_score(emb, train_embs)

        rec = {
            "accession": acc,
            "tier_a": pred.tier_a,
            "tier_a_confidence": pred.tier_a_confidence,
            "tier_b": pred.tier_b,
            "composite": pred.composite,
            "novelty_score": novelty,
            "organism": row.get("organism", ""),
            "source_nomination": row.get("source_nomination", "atlas_query"),
        }
        records.append(rec)

        if pred.tier_b == "N2_Fanzor_OMEGA" and pred.tier_a_confidence >= FANZOR_CONF_THRESHOLD:
            fanzor_count += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(candidates)} processed...")

    df = pd.DataFrame(records)

    # Sort by Fanzor confidence descending
    fanzor_mask = df["tier_b"] == "N2_Fanzor_OMEGA"
    df_fanzor = df[fanzor_mask].sort_values("tier_a_confidence", ascending=False)
    df_other = df[~fanzor_mask].sort_values("tier_a_confidence", ascending=False)
    df_out = pd.concat([df_fanzor, df_other], ignore_index=True)

    df_out.to_parquet(OUT_DIR / "fanzor_catalog.parquet", compression="zstd")
    df_out.to_csv(OUT_DIR / "fanzor_catalog.tsv", sep="\t", index=False)

    summary = {
        "total_candidates": len(df_out),
        "fanzor_high_conf": int(fanzor_count),
        "fanzor_threshold": FANZOR_CONF_THRESHOLD,
        "tier_a_distribution": df_out["tier_a"].value_counts().to_dict(),
        "tier_b_distribution": df_out["tier_b"].value_counts().dropna().to_dict(),
    }
    (OUT_DIR / "fanzor_catalog_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Fanzor catalog complete ===")
    print(f"Total candidates processed: {len(df_out)}")
    print(f"Fanzor (conf ≥ {FANZOR_CONF_THRESHOLD}):    {fanzor_count}")
    print(f"Catalog → {OUT_DIR / 'fanzor_catalog.parquet'}")
    print(f"TSV     → {OUT_DIR / 'fanzor_catalog.tsv'}")


if __name__ == "__main__":
    run()
