"""Evidence aggregation: weighted vote across all sources → EvidenceRecord.

Aggregation logic:
  1. For each protein, collect all evidence rows from all source Parquets.
  2. Compute weighted vote for Tier A: sum(evidence_weight × vote) per class.
  3. Confidence = top_class_score / total_score.
  4. contradiction_flag = True if ≥2 sources vote for different Tier A classes.
  5. reviewer_action:
       confidence ≥ 0.75 AND contradiction_flag == False → auto_accept
       0.50 ≤ confidence < 0.75 OR contradiction → manual_review
       confidence < 0.50 → discard (unless in foundational_systems.yaml)

Special rule: if IS110-family signal is present (TnPedia + foundational composite),
override any InterPro CL0219→DSB_NUCLEASE inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from importlib.resources import files as pkg_files

TIER_A_CLASSES = ["DSB_NUCLEASE", "DSB_FREE_TRANSEST_RECOMBINASE", "TRANSPOSASE"]

# Sources for the IS110 composite override rule
IS110_COMPOSITE_SOURCES = {"TnPedia_ISfinder", "TnPedia_curated", "Foundational_systems_v0.6.0"}

# High-authority sources — at least one must be present for a protein to enter
# the gold set (auto_accept or manual_review).  Pfam-whitelist and InterPro-clan
# rows are domain annotations only; they can corroborate but cannot alone qualify.
HIGH_AUTHORITY_SOURCES: frozenset[str] = frozenset({
    "M-CSA",
    "Foundational_systems_v0.6.0",
    "CRISPRCasdb",
    "Rhea",
    "UniProt_features",
    "TnPedia_ISfinder",
    "TnPedia_curated",
})

# Database-family mapping for n_sources deduplication.
# Multiple evidence rows from the same underlying database count as ONE source.
_DB_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("Pfam_whitelist", "Pfam_whitelist"),
    ("InterPro_clan",  "InterPro_clan"),
    ("InterPro",       "InterPro_clan"),   # interpro.py direct rows
    ("TnPedia",        "TnPedia"),
    ("M-CSA",          "M-CSA"),
    ("Foundational",   "Foundational"),
    ("CRISPRCasdb",    "CRISPRCasdb"),
    ("Rhea",           "Rhea"),
    ("UniProt_features", "UniProt_features"),
]


def _source_db_family(source: str) -> str:
    """Map a source string to its canonical database-family label."""
    for prefix, family in _DB_FAMILY_PREFIXES:
        if source.startswith(prefix):
            return family
    return source  # unknown source — keep as-is


@dataclass
class EvidenceRecord:
    uniprot_acc: str
    sources: dict = field(default_factory=dict)
    inferred_tier_a: str = "UNKNOWN"
    inferred_tier_b: str = "UNKNOWN"
    confidence_score: float = 0.0
    contradiction_flag: bool = False
    composite_architecture: bool = False
    reviewer_action: str = "discard"
    tier_a_votes: dict = field(default_factory=dict)
    n_sources: int = 0

    def to_dict(self) -> dict:
        return {
            "uniprot_acc": self.uniprot_acc,
            "inferred_tier_a": self.inferred_tier_a,
            "inferred_tier_b": self.inferred_tier_b,
            "confidence_score": self.confidence_score,
            "contradiction_flag": self.contradiction_flag,
            "composite_architecture": self.composite_architecture,
            "reviewer_action": self.reviewer_action,
            "n_sources": self.n_sources,
        }


def load_all_evidence(evidence_dir: Path) -> pd.DataFrame:
    """Load and concatenate all per-source evidence Parquets."""
    files = list(evidence_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No evidence Parquets found in {evidence_dir}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "uniprot_acc" in df.columns and "inferred_tier_a" in df.columns:
            dfs.append(df)
        else:
            print(f"  SKIP {f.name}: missing required columns (has: {list(df.columns)})")
    if not dfs:
        raise ValueError(
            f"No valid evidence parquets found in {evidence_dir} "
            f"(all {len(files)} files were missing uniprot_acc or inferred_tier_a columns)"
        )
    return pd.concat(dfs, ignore_index=True)


def aggregate_protein(group: pd.DataFrame, acc: str) -> EvidenceRecord:
    """Aggregate all evidence rows for one protein into an EvidenceRecord."""
    rec = EvidenceRecord(uniprot_acc=acc)

    # Weighted vote for Tier A; track source DB families for deduplication
    votes: dict[str, float] = {c: 0.0 for c in TIER_A_CLASSES}
    db_families: set[str] = set()
    has_high_auth: bool = False

    for _, row in group.iterrows():
        cls = row.get("inferred_tier_a", "")
        wt = float(row.get("evidence_weight", 0.5))
        src = row.get("source", "")
        if cls in votes:
            votes[cls] += wt
        rec.sources[src] = row.get("mcsa_id") or row.get("rhea_id") or src
        db_families.add(_source_db_family(src))
        if src in HIGH_AUTHORITY_SOURCES:
            has_high_auth = True

    # n_sources = number of distinct databases (not rows) — prevents Pfam domains
    # from inflating the source count and trivially satisfying multi-source rules.
    rec.n_sources = len(db_families)

    rec.tier_a_votes = votes
    total = sum(votes.values())
    if total == 0:
        return rec

    best = max(votes, key=votes.get)
    rec.confidence_score = round(votes[best] / total, 4)
    rec.inferred_tier_a = best

    # IS110 composite override: if any IS110-class source is present and
    # its inferred_tier_a is DSB_FREE, override InterPro CL0219→DSB_NUCLEASE
    is110_rows = group[
        group["source"].isin(IS110_COMPOSITE_SOURCES) &
        (group["inferred_tier_a"] == "DSB_FREE_TRANSEST_RECOMBINASE")
    ]
    if len(is110_rows) > 0:
        rec.inferred_tier_a = "DSB_FREE_TRANSEST_RECOMBINASE"
        rec.composite_architecture = bool(
            is110_rows["composite_architecture"].any()
            if "composite_architecture" in is110_rows.columns else False
        )

    # Tier B: take the tier_b from the highest-weight source
    if "inferred_tier_b" in group.columns:
        tier_b_rows = group[group["inferred_tier_b"].notna()].sort_values(
            "evidence_weight", ascending=False
        )
        if len(tier_b_rows):
            rec.inferred_tier_b = tier_b_rows.iloc[0]["inferred_tier_b"]

    # Composite flag
    if "composite_architecture" in group.columns:
        rec.composite_architecture = bool(group["composite_architecture"].any())

    # Contradiction flag
    active_classes = [c for c, v in votes.items() if v > 0]
    rec.contradiction_flag = len(active_classes) > 1

    # Reviewer action
    # Gate 0: must have at least one high-authority source to enter the gold set.
    # Proteins with only Pfam-whitelist / InterPro-clan annotations are domain
    # predictions, not curated mechanism evidence — exclude from training labels.
    if not has_high_auth:
        rec.reviewer_action = "discard"
    elif rec.confidence_score >= 0.75 and not rec.contradiction_flag:
        rec.reviewer_action = "auto_accept"
    elif rec.confidence_score >= 0.50 or rec.contradiction_flag:
        rec.reviewer_action = "manual_review"
    else:
        rec.reviewer_action = "discard"

    return rec


def load_atlas_protein_accessions(
    duckdb_path: str = "/data/graphs/atlas.duckdb",
) -> set[str]:
    """Return all UniProt accessions in the ATLAS (to restrict aggregation)."""
    import duckdb  # optional atlas dependency — lazy import
    con = duckdb.connect(duckdb_path, read_only=True)
    accs = set(con.execute("SELECT accession FROM nodes_protein").fetchdf()["accession"])
    con.close()
    return accs


def main(
    evidence_dir: Path = Path("/data/labels/evidence"),
    output: Path = Path("/data/labels/mechanism_labels_raw.parquet"),
    atlas_db: str = "/data/graphs/atlas.duckdb",
    auto_accept_threshold: float = 0.75,
) -> None:
    print("Loading all evidence sources...")
    evidence = load_all_evidence(evidence_dir)
    print(f"  Total evidence rows: {len(evidence):,}")
    print(f"  Unique UniProt accessions: {evidence['uniprot_acc'].nunique():,}")

    # Restrict to proteins in ATLAS (10,000-protein Paper 1 catalog)
    try:
        atlas_accs = load_atlas_protein_accessions(atlas_db)
        evidence = evidence[evidence["uniprot_acc"].isin(atlas_accs)]
        print(f"  After ATLAS filter: {len(evidence):,} rows, "
              f"{evidence['uniprot_acc'].nunique():,} proteins")
    except Exception as exc:
        print(f"  WARN: ATLAS filter failed ({exc}); proceeding without filter")

    # Aggregate per protein
    records = []
    for acc, group in evidence.groupby("uniprot_acc"):
        rec = aggregate_protein(group, str(acc))
        records.append(rec.to_dict())

    df = pd.DataFrame(records)

    # Summary
    print(f"\nAggregated {len(df):,} proteins")
    print("\nReviewer action distribution:")
    print(df["reviewer_action"].value_counts().to_string())
    gold = df[df["reviewer_action"].isin(["auto_accept", "manual_review"])]
    print(f"\nGold-set proteins (auto_accept + manual_review): {len(gold):,}")
    print("\nTier A distribution (gold set):")
    print(gold["inferred_tier_a"].value_counts().to_string())
    print("\nTier A distribution (all, including discard):")
    print(df["inferred_tier_a"].value_counts().to_string())
    n_auto = (df["reviewer_action"] == "auto_accept").sum()
    n_review = (df["reviewer_action"] == "manual_review").sum()
    n_composite = df["composite_architecture"].sum()
    print(f"\nAuto-accepted: {n_auto} | Manual review: {n_review} | Composite: {n_composite}")
    print(f"n_sources distribution (gold set):")
    print(gold["n_sources"].value_counts().sort_index().to_string())

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"\nWrote {len(df):,} aggregated evidence records -> {output}")


if __name__ == "__main__":
    main()
