"""CRISPRCasdb evidence reuse from Paper 1 (GENOME-ATLAS v0.6.0).

Paper 1 already ingested CRISPRCasdb April 2026 snapshot into two Parquet files:
  /data/processed/crisprcasdb_systems.parquet
  /data/processed/crisprcasdb_proteins.parquet

Columns in proteins parquet (verified 2026-05 from actual data):
  system_name, protein_name, uniprot_acc, role

Mechanism mapping is per (system_name, protein_name), NOT by crispr_type field.
The proteins parquet contains 19 rows spanning ALL mechanism classes — NOT just
CRISPR nucleases. Explicit per-protein mapping is required (see PROTEIN_MAP below).

Non-catalytic regulatory subunits (TnsC, TniQ, Cascade_TnsA, Prime_editor) are
excluded — they are structural/ATP-hydrolysis components, not the catalytic proteins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# (system_name, protein_name) -> (tier_a, tier_b, composite_architecture, evidence_weight)
# Verified against CRISPRCasdb April 2026 snapshot and primary literature (2026-05).
PROTEIN_MAP: dict[tuple[str, str], tuple[str, str, bool, float]] = {
    # ── CRISPR-Cas nucleases ──────────────────────────────────────────────
    ("SpCas9", "Cas9"): ("DSB_NUCLEASE", "N1_CRISPR_Cas", False, 0.9),
    ("Cas12a", "Cas12a"): ("DSB_NUCLEASE", "N1_CRISPR_Cas", False, 0.9),
    ("Cas12f", "Cas12f"): ("DSB_NUCLEASE", "N1_CRISPR_Cas", False, 0.9),
    # Cas12k (CAST-V-K) retains RuvC+Nuc domain architecture; classified as
    # DSB_NUCLEASE despite attenuated nuclease activity in CAST context.
    ("CAST-V-K", "Cas12k"): ("DSB_NUCLEASE", "N1_CRISPR_Cas", False, 0.85),
    # ── OMEGA/Fanzor nucleases ────────────────────────────────────────────
    # Fanzor proteins (IS200/IS605 TnpB-related) are RuvC-fold OMEGA nucleases.
    ("SpuFz1_Fanzor", "SpuFz1"): ("DSB_NUCLEASE", "N2_Fanzor_OMEGA", False, 0.9),
    ("MmeFz2_Fanzor", "MmeFz2"): ("DSB_NUCLEASE", "N2_Fanzor_OMEGA", False, 0.9),
    # ── IS110 composite transesterase recombinase (THE composite case) ────
    # PF01548 (RuvC-fold DEDD) + PF02371 (serine Tnp): transesterification,
    # NOT hydrolysis. Overrides CL0219 clan default.
    ("IS621_bridge_recombinase", "IS621_recombinase"): (
        "DSB_FREE_TRANSEST_RECOMBINASE",
        "B3_Programmable_Recombinase",
        True,
        0.9,
    ),
    # ── Site-specific recombinases ────────────────────────────────────────
    # Cre: tyrosine recombinase (loxP sites); Bxb1: large serine recombinase.
    # Both are in B1_Site_Specific_Recombinase per label_taxonomy.yaml v1.0.0.
    ("Cre_recombinase", "Cre"): ("DSB_FREE_TRANSEST_RECOMBINASE", "B1_Site_Specific_Recombinase", False, 0.9),
    ("Bxb1_integrase", "Bxb1"): ("DSB_FREE_TRANSEST_RECOMBINASE", "B1_Site_Specific_Recombinase", False, 0.9),
    # eePASSIGE: evolved IS110-family recombinase (engineered; no UniProt accession).
    # Included in map for completeness but will be filtered by uniprot_acc check.
    ("eePASSIGE", "eePASSIGE"): ("DSB_FREE_TRANSEST_RECOMBINASE", "B1_Site_Specific_Recombinase", False, 0.9),
    # ── DDE transposases ──────────────────────────────────────────────────
    ("Tn5_transposase", "Tn5"): ("TRANSPOSASE", "T1_DDE_Transposase", False, 0.9),
    # TnsB in CAST systems is a Tn7-family DDE transposase (the integration motor).
    ("CAST-V-K", "TnsB"): ("TRANSPOSASE", "T1_DDE_Transposase", False, 0.85),
    ("CAST-I-F_evoCAST", "TnsB"): ("TRANSPOSASE", "T1_DDE_Transposase", False, 0.85),
}

# Non-catalytic subunits — exclude from labeled evidence.
# TnsC: ATP-dependent DNA translocase; TniQ: DNA-targeting; Cascade_TnsA: guide.
# PE2 (prime editor): engineered fusion, no single-enzyme UniProt accession.
SKIP_PROTEIN_NAMES = frozenset({"TnsC", "TniQ", "Cascade_TnsA", "PE2"})


def main(
    proteins_path: Path = Path("/data/processed/crisprcasdb_proteins.parquet"),
    output: Path = Path("/data/labels/evidence/crisprcasdb.parquet"),
) -> None:
    proteins = pd.read_parquet(proteins_path)
    print(f"CRISPRCasdb proteins parquet: {len(proteins):,} rows, columns={list(proteins.columns)}")

    # Normalise the UniProt accession column name — actual column is
    # 'uniprot_accession' in the Paper 1 deposit (not 'uniprot_acc').
    if "uniprot_accession" in proteins.columns and "uniprot_acc" not in proteins.columns:
        proteins = proteins.rename(columns={"uniprot_accession": "uniprot_acc"})

    rows: list[dict] = []
    skipped_no_map = []
    for _, r in proteins.iterrows():
        sys_name = str(r.get("system_name") or "")
        prot_name = str(r.get("protein_name") or "")

        # Skip non-catalytic regulatory proteins
        if prot_name in SKIP_PROTEIN_NAMES:
            continue

        key = (sys_name, prot_name)
        mapping = PROTEIN_MAP.get(key)
        if mapping is None:
            skipped_no_map.append(key)
            continue

        tier_a, tier_b, composite, weight = mapping
        rows.append(
            {
                "source": "CRISPRCasdb",
                "uniprot_acc": r.get("uniprot_acc"),
                "system_name": sys_name,
                "protein_name": prot_name,
                "role": r.get("role"),
                "inferred_tier_a": tier_a,
                "inferred_tier_b": tier_b,
                "composite_architecture": composite,
                "evidence_weight": weight,
            }
        )

    if skipped_no_map:
        print(f"  WARNING: {len(skipped_no_map)} (system, protein) pairs have no mapping:")
        for k in skipped_no_map:
            print(f"    {k}")

    df = pd.DataFrame(rows)
    # Drop rows with no UniProt accession (engineered proteins, CAST without IDs)
    before = len(df)
    df = df[df["uniprot_acc"].notna() & (df["uniprot_acc"].astype(str) != "")]
    print(f"  {before - len(df)} rows dropped (no UniProt accession)")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} CRISPRCasdb evidence rows -> {output}")
    print(df.groupby("inferred_tier_a").size().to_string())

    # Verify IS621 composite case
    is621_rows = df[df["system_name"] == "IS621_bridge_recombinase"]
    if len(is621_rows):
        row = is621_rows.iloc[0]
        assert row["inferred_tier_a"] == "DSB_FREE_TRANSEST_RECOMBINASE", (
            f"IS621 must be DSB_FREE_TRANSEST_RECOMBINASE, got {row['inferred_tier_a']}"
        )
        assert row["composite_architecture"] is True
        print("IS621 composite case: DSB_FREE_TRANSEST_RECOMBINASE, composite=True ✓")


if __name__ == "__main__":
    main()
