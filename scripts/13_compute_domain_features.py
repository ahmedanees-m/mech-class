"""Compute Pfam domain features F_domain (26-dim).

Builds a binary feature vector over the 23 Pfam families from the GENOME-ATLAS
whitelist v1.2.0 plus 3 composite-architecture flags:
  - IS110 composite flag (PF01548 AND PF02371 both present)
  - Editor fusion flag (PF14739 Cas9-HNH AND PF00078 reverse transcriptase)
  - Single-domain flag (exactly one major catalytic domain)

Domain annotations come from ATLAS DuckDB HAS_DOMAIN edges (not interpro.parquet).
This guarantees all 572 labeled proteins have domain coverage since they are all
in the ~10,000-protein ATLAS.

Spot-check: IS110 composite proteins should show dom_{PF01548_idx}=1 AND
dom_{PF02371_idx}=1, is110_composite=1.

Run via:
    docker run --rm \\
        -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "
            SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps;
            pip install -e . --quiet;
            python scripts/13_compute_domain_features.py"

Expected output:
  /data/features/domain/F_domain.parquet
    columns: uniprot_acc, dom_0 .. dom_25 (26 total), is110_composite
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from mech_class.features.domain import (
    build_domain_feature_matrix,
    get_pfam_accessions,
    RUVC_DEDD_PF,
    TNP_SERINE_PF,
)

LABELS   = Path("/data/labels/mechanism_labels_final.parquet")
ATLAS_DB = "/data/graphs/atlas.duckdb"
OUT      = Path("/data/features/domain/F_domain.parquet")
ACC_COL  = "uniprot_acc"

OUT.parent.mkdir(parents=True, exist_ok=True)


def run() -> None:
    if not LABELS.exists():
        print(f"Labels not found at {LABELS}.")
        return

    labels     = pd.read_parquet(LABELS)
    accessions = labels[ACC_COL].tolist()
    wl_accs    = get_pfam_accessions()
    feat_dim   = len(wl_accs) + 3          # whitelist entries + 3 composite flags

    print(f"Building F_domain for {len(accessions)} proteins")
    print(f"Pfam whitelist: {len(wl_accs)} families + 3 composite flags = {feat_dim} dims")

    # Query ATLAS DuckDB for all HAS_DOMAIN edges to target proteins
    matrix = build_domain_feature_matrix(accessions, duckdb_path=ATLAS_DB)
    print(f"Domain matrix shape: {matrix.shape}")

    # Build output DataFrame
    records = []
    for i, acc in enumerate(accessions):
        rec = {ACC_COL: acc}
        vec = matrix[i]
        for j, v in enumerate(vec):
            rec[f"dom_{j}"] = float(v)
        # IS110 composite flag lives at index len(wl_accs) (first composite flag)
        n = len(wl_accs)
        rec["is110_composite"] = bool(vec[n] > 0.5)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(OUT, compression="zstd")

    n_composite = int(df["is110_composite"].sum())
    print(f"\n=== Domain features complete ===")
    print(f"Proteins:         {len(df)}")
    print(f"IS110 composite:  {n_composite}")
    print(f"Feature dim:      {feat_dim}")
    print(f"Output -> {OUT}")

    # Spot-check IS110 dual-domain proteins
    pf01548_idx = wl_accs.index(RUVC_DEDD_PF) if RUVC_DEDD_PF in wl_accs else None
    pf02371_idx = wl_accs.index(TNP_SERINE_PF) if TNP_SERINE_PF in wl_accs else None
    if pf01548_idx is not None and pf02371_idx is not None:
        is110_proteins = df[df["is110_composite"]][ACC_COL].tolist()
        print(f"\nIS110 composite proteins ({len(is110_proteins)}):")
        for acc in is110_proteins[:5]:
            row = df[df[ACC_COL] == acc].iloc[0]
            pf548 = row[f"dom_{pf01548_idx}"]
            pf371 = row[f"dom_{pf02371_idx}"]
            print(f"  {acc}: PF01548={pf548:.0f}, PF02371={pf371:.0f}")


if __name__ == "__main__":
    run()
