"""ATLAS domain-based protein labelling from Paper 1 graph (no API calls).

Joins nodes_protein → edges(HAS_DOMAIN) → nodes_domain in atlas.duckdb.
The nodes_domain table already carries mechanism_bucket from the Pfam whitelist
v1.2.0, so this is the most complete and reliable Pfam-level evidence source:
it covers all ~10,000 proteins in the ATLAS and requires no external API.

Two evidence rows are produced per protein-domain hit:
  1. Pfam whitelist (weight 0.6) — from nodes_domain.mechanism_bucket
  2. InterPro clan (weight 0.5) — from interpro.parquet, if the domain has a clan

Evidence weights match the hierarchy defined in LABEL_PROVENANCE.md.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ATLAS_DB = "/data/graphs/atlas.duckdb"
INTERPRO_P = "/data/labels/evidence/interpro.parquet"


def query_atlas_domains(db_path: str = ATLAS_DB) -> pd.DataFrame:
    """Return (uniprot_acc, pfam_acc, pfam_name, mechanism_bucket) for every
    protein-domain pair in the ATLAS."""
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT
            p.accession          AS uniprot_acc,
            d.accession          AS pfam_acc,
            d.name               AS pfam_name,
            d.mechanism_bucket   AS mechanism_bucket
        FROM nodes_protein  p
        JOIN edges          e ON p.id = e.source_id
                              AND e.edge_type = 'HAS_DOMAIN'
        JOIN nodes_domain   d ON d.id = e.target_id
        WHERE d.mechanism_bucket IS NOT NULL
          AND d.mechanism_bucket != ''
    """).fetchdf()
    con.close()
    return df


def query_composite_accs(db_path: str = ATLAS_DB) -> set[str]:
    """Return UniProt accessions of IS110-family bridge recombinases in the ATLAS.

    IS110-family proteins carry a genuine dual-domain composite architecture:
      PF01548 (DEDD_Tnp_IS110) — RuvC-fold N-terminal catalytic domain
      PF02371 (Transposase_20)  — serine-Tnp C-terminal domain

    Both domains are required for the bridge-recombinase transesterification
    mechanism (Hiraizumi et al. 2024 Nature; Vaysset et al. 2025 Nat Microbiol).
    Proteins carrying BOTH Pfam families in the same polypeptide are flagged
    composite_architecture=True.

    NOTE: PF07282 (Cas12f1-like_TNB / TnpB) is intentionally excluded.
    TnpB is a single-domain RNA-guided nuclease; its transposon-association
    is at the element level, not the protein level. The plan's composite
    definition (§0.2 / label_taxonomy.yaml) requires two catalytic modules
    of distinct evolutionary origin within the same polypeptide.
    """
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT p.accession AS uniprot_acc
        FROM nodes_protein p
        JOIN edges e1 ON p.id = e1.source_id AND e1.edge_type = 'HAS_DOMAIN'
        JOIN nodes_domain d1 ON d1.id = e1.target_id AND d1.accession = 'PF01548'
        JOIN edges e2 ON p.id = e2.source_id AND e2.edge_type = 'HAS_DOMAIN'
        JOIN nodes_domain d2 ON d2.id = e2.target_id AND d2.accession = 'PF02371'
    """).fetchdf()
    con.close()
    return set(df["uniprot_acc"])


def main(
    atlas_db: str = ATLAS_DB,
    interpro_path: Path = Path(INTERPRO_P),
    output: Path = Path("/data/labels/evidence/atlas_domain_evidence.parquet"),
) -> None:
    print("Querying ATLAS domain-protein graph...")
    hits = query_atlas_domains(atlas_db)
    print(f"  {len(hits):,} protein-domain hits, {hits['uniprot_acc'].nunique():,} unique proteins")

    # Identify IS110-family composite proteins (PF01548 + PF02371 co-occurrence).
    # PF07282 TnpB is intentionally excluded — single-domain nuclease, not composite.
    print("Identifying IS110 dual-domain composite proteins (PF01548 + PF02371)...")
    composite_accs = query_composite_accs(atlas_db)
    print(f"  {len(composite_accs):,} composite proteins identified")

    rows: list[dict] = []

    # ── Layer 1: Pfam whitelist evidence (weight 0.6) ────────────────────
    for _, r in hits.iterrows():
        rows.append(
            {
                "source": "Pfam_whitelist_v1.2.0_ATLAS",
                "uniprot_acc": r["uniprot_acc"],
                "pfam_acc": r["pfam_acc"],
                "pfam_name": r["pfam_name"],
                "inferred_tier_a": r["mechanism_bucket"],
                "composite_architecture": r["uniprot_acc"] in composite_accs,
                "evidence_weight": 0.6,
            }
        )

    # ── Layer 2: InterPro clan evidence (weight 0.5) ─────────────────────
    # IMPORTANT: Only include clan evidence that AGREES with the Pfam whitelist
    # tier_a for the same domain. InterPro clans (e.g. CL0219 RuvC-like) group
    # structurally similar domains regardless of mechanism — this causes spurious
    # contradictions when DDE transposases are in the same clan as RuvC nucleases.
    # Clan evidence adds CONFIRMATORY weight, not contradictory weight.
    if interpro_path.exists():
        interpro = pd.read_parquet(interpro_path)
        # interpro.parquet: pfam_acc, clan_acc, inferred_tier_a
        clan_map = interpro[interpro["inferred_tier_a"] != "UNKNOWN"].set_index("pfam_acc")["inferred_tier_a"].to_dict()
        clan_hits = hits[hits["pfam_acc"].isin(clan_map)].copy()
        n_total = len(clan_hits)
        added = 0
        skipped_contradictions = 0
        for _, r in clan_hits.iterrows():
            clan_tier_a = clan_map[r["pfam_acc"]]
            pfam_tier_a = r["mechanism_bucket"]
            # Only add clan evidence if it agrees with Pfam whitelist tier_a
            if clan_tier_a != pfam_tier_a:
                skipped_contradictions += 1
                continue
            rows.append(
                {
                    "source": "InterPro_clan_ATLAS",
                    "uniprot_acc": r["uniprot_acc"],
                    "pfam_acc": r["pfam_acc"],
                    "pfam_name": r["pfam_name"],
                    "inferred_tier_a": clan_tier_a,
                    "composite_architecture": r["uniprot_acc"] in composite_accs,
                    "evidence_weight": 0.5,
                }
            )
            added += 1
        print(
            f"  {n_total:,} clan hits: {added:,} added (agreement), "
            f"{skipped_contradictions:,} skipped (clan contradicts Pfam whitelist)"
        )
    else:
        print(f"  interpro.parquet not found at {interpro_path}; skipping clan layer")

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} atlas domain evidence rows -> {output}")
    print(f"Distinct proteins: {df['uniprot_acc'].nunique():,}")
    print("\nTier A distribution:")
    print(df.groupby("inferred_tier_a")["uniprot_acc"].nunique().to_string())


if __name__ == "__main__":
    main()
