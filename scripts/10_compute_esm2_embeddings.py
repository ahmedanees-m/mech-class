"""Step 8 — Verify ESM-2 embeddings are available from Paper 1 (Week 4).

ESM-2 150M (640-dim) embeddings were computed in Paper 1 and stored in
/data/embeddings/esm2_150M_v6.parquet (columns: accession, embedding, seq_length).
This script verifies coverage for all uniprot_acc in mechanism_labels_final.parquet
and reports any gaps.

If gaps exist (new accessions not in Paper 1 atlas), they are written to
/data/labels/esm2_missing.parquet for targeted re-embedding.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/10_compute_esm2_embeddings.py"

Expected output:
  Coverage report printed to stdout
  /data/labels/esm2_missing.parquet  (empty if 100% coverage)
  /data/features/seq/F_seq.parquet   (640-dim embeddings for all labeled proteins)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = Path("/data/labels/mechanism_labels_final.parquet")
ESM2_PARQUET = Path("/data/embeddings/esm2_150M_v6.parquet")
MISSING_OUT = Path("/data/labels/esm2_missing.parquet")
OUT = Path("/data/features/seq/F_seq.parquet")

ESM2_DIM = 640
ACC_COL = "uniprot_acc"          # column name in labels parquet


def run() -> None:
    if not LABELS.exists():
        print(f"Labels not found at {LABELS}. Run 08_ingest_curator_decisions.py first.")
        return

    labels = pd.read_parquet(LABELS)
    label_accs = list(labels[ACC_COL])
    label_acc_set = set(label_accs)
    print(f"Labeled proteins: {len(label_accs)}")

    # Load Paper 1 ESM-2 embeddings (column 'accession' in this parquet)
    esm2 = pd.read_parquet(ESM2_PARQUET)
    esm2_acc_set = set(esm2["accession"])

    covered = label_acc_set & esm2_acc_set
    missing = label_acc_set - esm2_acc_set
    print(f"ESM-2 v6 parquet: {len(esm2)} proteins")
    print(f"Covered:          {len(covered)}")
    print(f"Missing:          {len(missing)}")

    if missing:
        miss_df = labels[labels[ACC_COL].isin(missing)][[ACC_COL]].copy()
        miss_df.to_parquet(MISSING_OUT, compression="zstd")
        print(f"\n[WARNING] {len(missing)} proteins lack ESM-2 embeddings.")
        print(f"  Written to {MISSING_OUT}")
        print("  Re-embed using pen-stack/plm:0.1.0 with ESM-2 150M before continuing.")
        print(f"  Missing: {sorted(missing)}")
    else:
        pd.DataFrame(columns=[ACC_COL]).to_parquet(MISSING_OUT, compression="zstd")
        print("\nAll labeled proteins have ESM-2 embeddings.")

    # Build F_seq parquet: 640 named columns + uniprot_acc
    OUT.parent.mkdir(parents=True, exist_ok=True)
    esm2_idx = esm2.set_index("accession")
    rows = []
    for acc in label_accs:
        if acc in esm2_idx.index:
            vec = np.asarray(esm2_idx.loc[acc, "embedding"], dtype=np.float32)
        else:
            vec = np.zeros(ESM2_DIM, dtype=np.float32)
        rec = {ACC_COL: acc}
        for j, v in enumerate(vec):
            rec[f"seq_{j}"] = float(v)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, compression="zstd")
    print(f"\nWrote F_seq: {df.shape} -> {OUT}")
    print(f"  Columns: {ACC_COL} + seq_0 .. seq_{ESM2_DIM - 1}")


if __name__ == "__main__":
    run()
