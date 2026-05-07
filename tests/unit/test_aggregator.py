"""Unit tests for evidence aggregator — aggregate_protein(), IS110 override.

Uses the actual API: aggregate_protein(group: pd.DataFrame, acc: str) → EvidenceRecord.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mech_class.evidence.aggregator import (
    HIGH_AUTHORITY_SOURCES,
    IS110_COMPOSITE_SOURCES,
    EvidenceRecord,
    aggregate_protein,
    load_all_evidence,
)

_IS110_SRC = next(iter(IS110_COMPOSITE_SOURCES))  # e.g. "TnPedia_ISfinder"


def _make_group(rows: list[dict]) -> pd.DataFrame:
    """Build a per-protein evidence DataFrame with required columns."""
    defaults = {
        "uniprot_acc": "P_TEST",
        "inferred_tier_a": "DSB_NUCLEASE",
        "evidence_weight": 0.5,
        "source": "M-CSA",
        "inferred_tier_b": None,
        "composite_architecture": False,
    }
    filled = [{**defaults, **r} for r in rows]
    return pd.DataFrame(filled)


# ── Weighted vote / Tier-A inference ─────────────────────────────────────────


class TestWeightedVote:
    def test_single_source_returns_correct_tier(self):
        group = _make_group([{"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 1.0}])
        rec = aggregate_protein(group, "P001")
        assert rec.inferred_tier_a == "DSB_NUCLEASE"

    def test_majority_wins(self):
        group = _make_group(
            [
                {"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 1.0, "source": "M-CSA"},
                {"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 0.8, "source": "Rhea"},
                {"inferred_tier_a": "TRANSPOSASE", "evidence_weight": 0.6, "source": "CRISPRCasdb"},
            ]
        )
        rec = aggregate_protein(group, "P002")
        assert rec.inferred_tier_a == "DSB_NUCLEASE"

    def test_confidence_in_range(self):
        group = _make_group([{"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 0.9}])
        rec = aggregate_protein(group, "P003")
        assert 0.0 <= rec.confidence_score <= 1.0

    def test_contradiction_flag_raised(self):
        group = _make_group(
            [
                {"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 1.0, "source": "M-CSA"},
                {"inferred_tier_a": "TRANSPOSASE", "evidence_weight": 1.0, "source": "Rhea"},
            ]
        )
        rec = aggregate_protein(group, "P004")
        assert rec.contradiction_flag is True

    def test_no_contradiction_single_class(self):
        group = _make_group(
            [
                {"inferred_tier_a": "DSB_NUCLEASE", "source": "M-CSA"},
                {"inferred_tier_a": "DSB_NUCLEASE", "source": "Rhea"},
            ]
        )
        rec = aggregate_protein(group, "P005")
        assert rec.contradiction_flag is False

    def test_empty_group_returns_discard(self):
        """Zero weight rows → confidence 0 → discard."""
        group = _make_group([{"evidence_weight": 0.0, "source": "InterPro"}])
        rec = aggregate_protein(group, "P006")
        assert rec.reviewer_action == "discard"


# ── IS110 override logic ──────────────────────────────────────────────────────


class TestIS110Override:
    def test_override_fires_with_is110_source(self):
        """TnPedia source + inferred DSB_FREE → override InterPro DSB_NUCLEASE."""
        group = _make_group(
            [
                {
                    "inferred_tier_a": "DSB_NUCLEASE",
                    "evidence_weight": 0.5,
                    "source": "InterPro",
                    "composite_architecture": True,
                },
                {
                    "inferred_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
                    "evidence_weight": 0.7,
                    "source": _IS110_SRC,
                    "composite_architecture": True,
                },
            ]
        )
        rec = aggregate_protein(group, "IS110_001")
        assert rec.inferred_tier_a == "DSB_FREE_TRANSEST_RECOMBINASE", (
            f"IS110 override should fire; got {rec.inferred_tier_a!r}"
        )

    def test_no_override_without_is110_source(self):
        """Without a TnPedia/Foundational source, no IS110 override."""
        group = _make_group(
            [
                {"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 0.9, "source": "M-CSA"},
            ]
        )
        rec = aggregate_protein(group, "P007")
        assert rec.inferred_tier_a == "DSB_NUCLEASE"

    def test_no_override_when_is110_source_votes_nuclease(self):
        """IS110 override only fires when the IS110-source row votes DSB_FREE."""
        group = _make_group(
            [
                {
                    "inferred_tier_a": "DSB_NUCLEASE",
                    "evidence_weight": 0.7,
                    "source": _IS110_SRC,
                    "composite_architecture": False,
                },
            ]
        )
        rec = aggregate_protein(group, "P008")
        # IS110-source votes DSB_NUCLEASE — override should NOT fire
        assert rec.inferred_tier_a == "DSB_NUCLEASE"


# ── Reviewer action rules ────────────────────────────────────────────────────


class TestReviewerAction:
    def test_auto_accept_high_confidence_no_contradiction(self):
        group = _make_group(
            [
                {"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 1.0, "source": "M-CSA"},
                {"inferred_tier_a": "DSB_NUCLEASE", "evidence_weight": 0.8, "source": "Rhea"},
            ]
        )
        rec = aggregate_protein(group, "P_AUTO")
        assert rec.reviewer_action == "auto_accept"

    def test_discard_for_no_high_authority_source(self):
        """InterPro alone (not in HIGH_AUTHORITY_SOURCES) → discard."""
        group = _make_group(
            [
                {"inferred_tier_a": "TRANSPOSASE", "evidence_weight": 1.0, "source": "InterPro"},
            ]
        )
        rec = aggregate_protein(group, "P_DISCARD")
        assert rec.reviewer_action == "discard"

    def test_accession_propagated(self):
        group = _make_group([{"inferred_tier_a": "DSB_NUCLEASE", "source": "M-CSA"}])
        rec = aggregate_protein(group, "MY_ACCESSION")
        assert rec.uniprot_acc == "MY_ACCESSION"


# ── EvidenceRecord.to_dict() ──────────────────────────────────────────────────


class TestEvidenceRecordToDict:
    def test_to_dict_has_required_keys(self):
        rec = EvidenceRecord(uniprot_acc="TEST_ACC")
        d = rec.to_dict()
        required = {
            "uniprot_acc",
            "inferred_tier_a",
            "inferred_tier_b",
            "confidence_score",
            "contradiction_flag",
            "composite_architecture",
            "reviewer_action",
            "n_sources",
        }
        assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"

    def test_to_dict_values_are_serializable(self):
        """All values in to_dict() must be JSON-serializable primitives."""
        import json

        rec = EvidenceRecord(uniprot_acc="P001")
        d = rec.to_dict()
        # Should not raise
        json.dumps(d)


# ── load_all_evidence ─────────────────────────────────────────────────────────


class TestLoadAllEvidence:
    def test_loads_single_parquet(self, tmp_path):
        """load_all_evidence reads parquets from a directory."""
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        df = pd.DataFrame(
            [
                {
                    "uniprot_acc": "P001",
                    "inferred_tier_a": "DSB_NUCLEASE",
                    "evidence_weight": 1.0,
                    "source": "M-CSA",
                }
            ]
        )
        df.to_parquet(ev_dir / "mcsa.parquet")
        result = load_all_evidence(ev_dir)
        assert len(result) == 1
        assert result.iloc[0]["uniprot_acc"] == "P001"

    def test_raises_on_empty_directory(self, tmp_path):
        """Empty evidence directory must raise FileNotFoundError."""
        ev_dir = tmp_path / "empty"
        ev_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No evidence Parquets"):
            load_all_evidence(ev_dir)

    def test_skips_parquet_missing_required_columns(self, tmp_path):
        """Parquets without 'uniprot_acc' or 'inferred_tier_a' are skipped."""
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        # This parquet is missing the required columns — should be skipped
        bad_df = pd.DataFrame([{"wrong_col": "value"}])
        bad_df.to_parquet(ev_dir / "bad.parquet")
        # Good parquet
        good_df = pd.DataFrame(
            [
                {
                    "uniprot_acc": "P001",
                    "inferred_tier_a": "DSB_NUCLEASE",
                    "evidence_weight": 0.9,
                    "source": "M-CSA",
                }
            ]
        )
        good_df.to_parquet(ev_dir / "good.parquet")
        result = load_all_evidence(ev_dir)
        assert len(result) == 1  # only the good parquet

    def test_concatenates_multiple_parquets(self, tmp_path):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        for i, acc in enumerate(["P001", "P002"]):
            df = pd.DataFrame(
                [
                    {
                        "uniprot_acc": acc,
                        "inferred_tier_a": "TRANSPOSASE",
                        "evidence_weight": 0.8,
                        "source": "TnPedia_ISfinder",
                    }
                ]
            )
            df.to_parquet(ev_dir / f"source_{i}.parquet")
        result = load_all_evidence(ev_dir)
        assert len(result) == 2


# ── Aggregator constants ───────────────────────────────────────────────────────


class TestAggregatorConstants:
    def test_is110_composite_sources_nonempty(self):
        assert len(IS110_COMPOSITE_SOURCES) >= 2

    def test_tnpedia_in_is110_sources(self):
        assert "TnPedia_ISfinder" in IS110_COMPOSITE_SOURCES

    def test_high_authority_sources_include_mcsa(self):
        assert "M-CSA" in HIGH_AUTHORITY_SOURCES

    def test_high_authority_sources_include_tnpedia(self):
        assert "TnPedia_ISfinder" in HIGH_AUTHORITY_SOURCES
