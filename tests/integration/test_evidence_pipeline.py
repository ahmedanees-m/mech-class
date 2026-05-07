"""Integration tests for evidence pipeline (scripts 01-06).

These tests use synthetic in-memory data to verify the end-to-end aggregation
pipeline without requiring live API calls or the Docker environment.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mech_class.evidence.aggregator import main as aggregate_main

# Column names MUST match aggregator.load_all_evidence expectations:
#   uniprot_acc, inferred_tier_a, evidence_weight, source,
#   inferred_tier_b (optional), composite_architecture (optional)
SYNTHETIC_EVIDENCE = [
    # M-CSA (weight 1.0) — confident DSB_NUCLEASE
    {
        "uniprot_acc": "P001",
        "inferred_tier_a": "DSB_NUCLEASE",
        "evidence_weight": 1.0,
        "source": "M-CSA",
        "inferred_tier_b": "N1_CRISPR_Cas",
        "composite_architecture": False,
    },
    # Rhea corroboration
    {
        "uniprot_acc": "P001",
        "inferred_tier_a": "DSB_NUCLEASE",
        "evidence_weight": 0.8,
        "source": "Rhea",
        "inferred_tier_b": "N1_CRISPR_Cas",
        "composite_architecture": False,
    },
    # IS110 composite — InterPro says DSB_NUCLEASE (CL0219 clan), but TnPedia overrides
    {
        "uniprot_acc": "P002",
        "inferred_tier_a": "DSB_NUCLEASE",
        "evidence_weight": 0.5,
        "source": "InterPro",
        "inferred_tier_b": None,
        "composite_architecture": True,
    },
    {
        "uniprot_acc": "P002",
        "inferred_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
        "evidence_weight": 0.7,
        "source": "TnPedia_ISfinder",
        "inferred_tier_b": "B3_Programmable_Recombinase",
        "composite_architecture": True,
    },
    # Low confidence — single weak source → manual review or discard
    {
        "uniprot_acc": "P003",
        "inferred_tier_a": "TRANSPOSASE",
        "evidence_weight": 0.5,
        "source": "InterPro",
        "inferred_tier_b": "T1_DDE_Transposase",
        "composite_architecture": False,
    },
]


@pytest.fixture
def evidence_dir(tmp_path):
    ev_dir = tmp_path / "evidence"
    ev_dir.mkdir()
    df = pd.DataFrame(SYNTHETIC_EVIDENCE)
    df.to_parquet(ev_dir / "synthetic.parquet", compression="zstd")
    return ev_dir


def test_aggregate_produces_output(evidence_dir, tmp_path):
    out_path = tmp_path / "labels.parquet"
    # atlas_db=":memory:" → DuckDB finds no nodes_protein table → ATLAS filter skipped
    aggregate_main(evidence_dir=evidence_dir, output=out_path, atlas_db=":memory:")
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == 3  # P001, P002, P003


def test_p001_auto_accept(evidence_dir, tmp_path):
    out_path = tmp_path / "labels.parquet"
    aggregate_main(evidence_dir=evidence_dir, output=out_path, atlas_db=":memory:")
    df = pd.read_parquet(out_path)
    row = df[df["uniprot_acc"] == "P001"].iloc[0]
    assert row["inferred_tier_a"] == "DSB_NUCLEASE"
    assert row["reviewer_action"] == "auto_accept"


def test_p002_is110_override(evidence_dir, tmp_path):
    out_path = tmp_path / "labels.parquet"
    aggregate_main(evidence_dir=evidence_dir, output=out_path, atlas_db=":memory:")
    df = pd.read_parquet(out_path)
    row = df[df["uniprot_acc"] == "P002"].iloc[0]
    # IS110 override: TnPedia_ISfinder source + composite_architecture=True → DSB_FREE
    assert row["inferred_tier_a"] == "DSB_FREE_TRANSEST_RECOMBINASE", (
        f"IS110 composite override failed: got {row['inferred_tier_a']}"
    )


def test_p003_discard(evidence_dir, tmp_path):
    """P003 has only 'InterPro' as source, which is NOT a high-authority source.

    HIGH_AUTHORITY_SOURCES gate (added in aggregator v0.5): proteins without
    at least one high-authority source (M-CSA, Rhea, TnPedia, etc.) are discarded
    even with high confidence. InterPro domain annotation alone cannot qualify
    a protein for the gold set.
    """
    out_path = tmp_path / "labels.parquet"
    aggregate_main(evidence_dir=evidence_dir, output=out_path, atlas_db=":memory:")
    df = pd.read_parquet(out_path)
    row = df[df["uniprot_acc"] == "P003"].iloc[0]
    assert row["reviewer_action"] == "discard", (
        f"P003 (InterPro-only evidence) should be discarded by HIGH_AUTHORITY_SOURCES gate; "
        f"got {row['reviewer_action']!r}"
    )
