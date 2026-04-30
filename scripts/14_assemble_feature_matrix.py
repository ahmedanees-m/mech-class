"""Assemble full feature matrix from four channels.

Joins:
  F_seq        : ESM-2 150M embeddings  (640-dim)   from /data/features/seq/F_seq.parquet
  F_struct     : SaProt 650M embeddings (1280-dim)  from /data/features/struct/F_struct.parquet
  F_domain     : Pfam binary flags      (26-dim)    from /data/features/domain/F_domain.parquet
  F_active_site: Active-site geometry   (7-dim)     from /data/features/active_site/F_active_site.parquet

All four channels are joined on 'uniprot_acc'. Missing proteins in any channel
are zero-imputed (left outer join). The hold-out exclusion is verified: if any
of Q99ZW2 / Q46731 / O25753 slip in during the merge, the script aborts.

Writes:
  /data/features/fused/feature_matrix.parquet   (572 x 1953 feature cols + uniprot_acc)
  /data/features/fused/labels.parquet           (572 x label cols + uniprot_acc)
  /data/features/fused/feature_manifest.json

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
            python scripts/14_assemble_feature_matrix.py"
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS      = Path("/data/labels/mechanism_labels_final.parquet")
F_SEQ       = Path("/data/features/seq/F_seq.parquet")
F_STRUCT    = Path("/data/features/struct/F_struct.parquet")
F_DOMAIN    = Path("/data/features/domain/F_domain.parquet")
F_ACTIVE    = Path("/data/features/active_site/F_active_site.parquet")

OUT_DIR     = Path("/data/features/fused")
OUT_FEAT    = OUT_DIR / "feature_matrix.parquet"
OUT_LABELS  = OUT_DIR / "labels.parquet"
MANIFEST    = OUT_DIR / "feature_manifest.json"

ACC_COL     = "uniprot_acc"
HOLDOUT_SET = frozenset({"Q99ZW2", "Q46731", "O25753"})

# Expected dims for each channel
EXPECT_SEQ    = 640
EXPECT_STRUCT = 1280
EXPECT_DOMAIN = 26
EXPECT_ACTIVE = 7

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_channel(path: Path, prefix: str, n_expected: int) -> pd.DataFrame | None:
    """Load a channel parquet and validate column count."""
    if not path.exists():
        print(f"  [WARN] {path.name} not found; channel will be zero-filled.")
        return None
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    print(f"  {path.name}: {len(feat_cols)} feature cols (expected {n_expected})")
    if len(feat_cols) != n_expected:
        print(f"  [WARN] dim mismatch - got {len(feat_cols)}, expected {n_expected}")
    return df


def run() -> None:
    if not LABELS.exists():
        print(f"Labels not found at {LABELS}.")
        return

    labels = pd.read_parquet(LABELS)
    accessions = labels[ACC_COL].tolist()
    print(f"Assembling feature matrix for {len(accessions)} proteins")

    # --- Hold-out safety check -----------------------------------------------
    leaked = set(accessions) & HOLDOUT_SET
    if leaked:
        raise RuntimeError(
            f"Hold-out proteins leaked into labels: {leaked}. "
            f"Re-run 08_ingest_curator_decisions.py."
        )
    print(f"Hold-out check PASS (none of {HOLDOUT_SET} in labels).")

    # --- Load channels --------------------------------------------------------
    print("\nLoading channels:")
    seq_df    = _load_channel(F_SEQ,    "seq_",    EXPECT_SEQ)
    struct_df = _load_channel(F_STRUCT, "struct_", EXPECT_STRUCT)
    domain_df = _load_channel(F_DOMAIN, "dom_",    EXPECT_DOMAIN)
    active_df = _load_channel(F_ACTIVE, "as_",     EXPECT_ACTIVE)

    # --- Merge (left join on uniprot_acc) ------------------------------------
    df = pd.DataFrame({ACC_COL: accessions})

    def _merge(base: pd.DataFrame, other: pd.DataFrame | None, keep_cols: list[str]) -> pd.DataFrame:
        if other is None:
            return base
        cols = [ACC_COL] + [c for c in other.columns if c in keep_cols or
                            any(c.startswith(p) for p in ["seq_", "struct_", "dom_", "as_"])]
        return base.merge(other[[c for c in cols if c in other.columns]], on=ACC_COL, how="left")

    if seq_df is not None:
        seq_cols = [c for c in seq_df.columns if c.startswith("seq_")]
        df = df.merge(seq_df[[ACC_COL] + seq_cols], on=ACC_COL, how="left")

    if struct_df is not None:
        struct_cols = [c for c in struct_df.columns if c.startswith("struct_")]
        df = df.merge(struct_df[[ACC_COL] + struct_cols], on=ACC_COL, how="left")

    if domain_df is not None:
        dom_cols = [c for c in domain_df.columns if c.startswith("dom_")]
        df = df.merge(domain_df[[ACC_COL] + dom_cols], on=ACC_COL, how="left")

    if active_df is not None:
        as_cols  = [c for c in active_df.columns if c.startswith("as_")]
        df = df.merge(active_df[[ACC_COL] + as_cols], on=ACC_COL, how="left")

    # Zero-impute any NaN from missing proteins in any channel
    feat_cols = [c for c in df.columns if c != ACC_COL]
    df[feat_cols] = df[feat_cols].fillna(0.0)

    # --- Labels parquet -------------------------------------------------------
    label_out = labels[[
        ACC_COL, "inferred_tier_a", "inferred_tier_b",
        "composite_architecture", "reviewer_action", "confidence_score",
    ]].copy()
    # Friendly short aliases for training scripts
    label_out["tier_a"]         = label_out["inferred_tier_a"]
    label_out["tier_b"]         = label_out["inferred_tier_b"]
    label_out["composite_flag"] = label_out["composite_architecture"]

    # --- Write ----------------------------------------------------------------
    df.to_parquet(OUT_FEAT, compression="zstd")
    label_out.to_parquet(OUT_LABELS, compression="zstd")

    # --- Manifest -------------------------------------------------------------
    seq_f    = [c for c in feat_cols if c.startswith("seq_")]
    struct_f = [c for c in feat_cols if c.startswith("struct_")]
    dom_f    = [c for c in feat_cols if c.startswith("dom_")]
    as_f     = [c for c in feat_cols if c.startswith("as_")]
    total    = len(seq_f) + len(struct_f) + len(dom_f) + len(as_f)

    manifest = {
        "n_proteins"    : len(df),
        "total_features": total,
        "expected_total": EXPECT_SEQ + EXPECT_STRUCT + EXPECT_DOMAIN + EXPECT_ACTIVE,
        "channels": {
            "F_seq"        : {"dim": len(seq_f),    "source": str(F_SEQ)},
            "F_struct"     : {"dim": len(struct_f), "source": str(F_STRUCT)},
            "F_domain"     : {"dim": len(dom_f),    "source": str(F_DOMAIN)},
            "F_active_site": {"dim": len(as_f),     "source": str(F_ACTIVE)},
        },
        "holdouts_verified": sorted(HOLDOUT_SET),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    # --- Summary --------------------------------------------------------------
    print(f"\n=== Feature matrix assembled ===")
    print(f"Proteins:         {len(df)}")
    print(f"Total features:   {total}  (expected {EXPECT_SEQ+EXPECT_STRUCT+EXPECT_DOMAIN+EXPECT_ACTIVE})")
    print(f"  F_seq:          {len(seq_f)}")
    print(f"  F_struct:       {len(struct_f)}")
    print(f"  F_domain:       {len(dom_f)}")
    print(f"  F_active_site:  {len(as_f)}")
    print(f"\nfeature_matrix -> {OUT_FEAT}")
    print(f"labels         -> {OUT_LABELS}")
    print(f"manifest       -> {MANIFEST}")

    # Sanity check: no all-zero rows in seq or struct channels (ESM-2 should be 100%)
    if seq_f:
        n_zero_seq = int((df[seq_f].abs().sum(axis=1) == 0).sum())
        if n_zero_seq:
            print(f"\n[WARN] {n_zero_seq} proteins have all-zero F_seq (missing ESM-2)")
        else:
            print("\nF_seq completeness: 100% (no zero rows)")


if __name__ == "__main__":
    run()
