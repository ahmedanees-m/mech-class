"""Unit tests for mech_class.api helper functions (no model files required).

Tests cover the pure-function helpers in api.py:
  _build_feature_row()    - assembles 1-row feature DataFrame from ESM-2 + Pfam hits
  _download_models() - stub that raises RuntimeError (URL not yet configured)
  _fetch_pfam_hits()      - UniProt REST lookup (tested for error fallback only)
  Prediction.summary()    - string formatting

These tests do NOT require trained model files at /data/models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mech_class.api import (
    PFAM_WHITELIST,
    Prediction,
    _build_feature_row,
    _download_models,
    _fetch_pfam_hits,
)

# _build_feature_row


class TestBuildFeatureRow:
    """_build_feature_row(seq_emb, pfam_hits, feat_cols) -> 1-row pd.DataFrame."""

    def _make_feat_cols(self) -> list[str]:
        """Minimal feature column list: seq_0..639 + dom_0..25."""
        return [f"seq_{i}" for i in range(640)] + [f"dom_{i}" for i in range(26)]

    def test_output_is_dataframe_with_one_row(self):
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, [], feat_cols)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == feat_cols

    def test_seq_channel_filled_correctly(self):
        feat_cols = self._make_feat_cols()
        seq_emb = np.arange(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, [], feat_cols)
        for i in range(640):
            assert df.iloc[0][f"seq_{i}"] == pytest.approx(float(i))

    def test_dom_channel_zero_for_empty_pfam(self):
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, [], feat_cols)
        for i in range(26):
            assert df.iloc[0][f"dom_{i}"] == 0.0

    def test_known_pfam_sets_dom_bit(self):
        """PF01548 is dom_4 in PFAM_WHITELIST; passing it should set dom_4=1.0."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF01548"], feat_cols)
        assert df.iloc[0]["dom_4"] == 1.0, "PF01548 -> dom_4 should be 1.0"

    def test_is110_composite_flag_dom23(self):
        """PF01548 + PF02371 -> dom_23 (IS110 composite) = 1.0."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF01548", "PF02371"], feat_cols)
        assert df.iloc[0]["dom_23"] == 1.0

    def test_is110_composite_flag_dom23_requires_both(self):
        """Only PF01548 (no PF02371) -> dom_23 = 0.0."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF01548"], feat_cols)
        assert df.iloc[0]["dom_23"] == 0.0

    def test_single_domain_flag_dom25(self):
        """Exactly one whitelist Pfam -> dom_25 = 1.0."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF00589"], feat_cols)  # Phage_integrase, dom_13
        assert df.iloc[0]["dom_25"] == 1.0

    def test_two_whitelist_domains_clears_dom25(self):
        """Two whitelist Pfams -> dom_25 = 0.0."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF01548", "PF02371"], feat_cols)
        assert df.iloc[0]["dom_25"] == 0.0

    def test_unknown_pfam_does_not_set_any_dom_bit(self):
        """Unknown Pfam accession must not set any dom_0..22 bit."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.zeros(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF99999_FAKE"], feat_cols)
        for i in range(23):
            assert df.iloc[0][f"dom_{i}"] == 0.0, f"dom_{i} should be 0 for unknown Pfam"

    def test_dom24_always_zero(self):
        """dom_24 (editor fusion - reserved) must always be 0."""
        feat_cols = self._make_feat_cols()
        seq_emb = np.ones(640, dtype=np.float32)
        # Pass many Pfams - dom_24 must still be 0
        df = _build_feature_row(seq_emb, list(PFAM_WHITELIST), feat_cols)
        assert df.iloc[0]["dom_24"] == 0.0

    def test_partial_feat_cols_no_error(self):
        """If feat_cols only contains a subset of seq/dom columns, no KeyError."""
        feat_cols = ["seq_0", "seq_1", "dom_4", "dom_23"]
        seq_emb = np.arange(640, dtype=np.float32)
        df = _build_feature_row(seq_emb, ["PF01548", "PF02371"], feat_cols)
        assert len(df.columns) == 4
        assert df.iloc[0]["seq_0"] == 0.0
        assert df.iloc[0]["seq_1"] == 1.0
        assert df.iloc[0]["dom_4"] == 1.0
        assert df.iloc[0]["dom_23"] == 1.0


# Predictor.load() - error paths (no model files required)


class TestPredictorLoadErrorPaths:
    def test_raises_file_not_found_for_missing_dir(self, tmp_path):
        """Predictor.load() with non-existent dir must raise FileNotFoundError."""
        from mech_class.api import Predictor

        with pytest.raises(FileNotFoundError, match="Tier-A model not found"):
            Predictor.load(tmp_path / "does_not_exist")

    def test_raises_file_not_found_message_contains_hint(self, tmp_path):
        """The FileNotFoundError message must contain user guidance."""
        from mech_class.api import Predictor

        with pytest.raises(FileNotFoundError) as exc_info:
            Predictor.load(tmp_path / "no_models")
        msg = str(exc_info.value)
        assert "model_dir" in msg or "Predictor.load" in msg


# _download_models


class TestDownloadModels:
    def test_raises_runtime_error(self, tmp_path):
        """_download_models is a stub that must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not yet configured"):
            _download_models(tmp_path)

    def test_error_message_contains_instructions(self, tmp_path):
        """The RuntimeError message must tell users how to pass model_dir."""
        with pytest.raises(RuntimeError) as exc_info:
            _download_models(tmp_path)
        msg = str(exc_info.value)
        assert "model_dir" in msg or "Predictor.load" in msg


# _fetch_pfam_hits fallback


class TestFetchPfamHits:
    def test_returns_empty_list_on_invalid_accession(self):
        """Invalid accession -> 404 from UniProt -> empty list (no exception)."""
        result = _fetch_pfam_hits("INVALID_ACCESSION_XYZ", timeout=5)
        assert isinstance(result, list)
        assert result == []

    def test_returns_list_type_always(self):
        """Must always return a list, even on network error."""
        result = _fetch_pfam_hits("UNREACHABLE_HOST_99999", timeout=1)
        assert isinstance(result, list)


# Prediction model


class TestPredictionModel:
    def _make_pred(self, **kwargs) -> Prediction:
        defaults = {
            "accession": "P_TEST",
            "sequence_length": 300,
            "tier_a": "DSB_NUCLEASE",
            "tier_a_confidence": 0.95,
            "composite": False,
            "composite_prob": 0.05,
        }
        return Prediction(**{**defaults, **kwargs})

    def test_summary_contains_accession(self):
        pred = self._make_pred()
        assert "P_TEST" in pred.summary()

    def test_summary_contains_tier_a(self):
        pred = self._make_pred()
        assert "DSB_NUCLEASE" in pred.summary()

    def test_summary_composite_not_shown_when_false(self):
        pred = self._make_pred(composite=False, composite_prob=0.05)
        assert "COMPOSITE" not in pred.summary()

    def test_summary_composite_shown_when_true(self):
        pred = self._make_pred(composite=True, composite_prob=0.85)
        assert "COMPOSITE" in pred.summary()

    def test_confidence_property_alias(self):
        pred = self._make_pred(tier_a_confidence=0.88)
        assert pred.confidence == pytest.approx(0.88)

    def test_default_channels_used_empty(self):
        pred = self._make_pred()
        assert pred.channels_used == []

    def test_pfam_hits_default_empty(self):
        pred = self._make_pred()
        assert pred.pfam_hits == []

    def test_tier_b_none_by_default(self):
        pred = self._make_pred()
        assert pred.tier_b is None
        assert pred.tier_b_confidence is None

    def test_tier_b_confidence_when_set(self):
        pred = self._make_pred(tier_b="N1_CRISPR_Cas", tier_b_confidence=0.72)
        assert pred.tier_b == "N1_CRISPR_Cas"
        assert pred.tier_b_confidence == pytest.approx(0.72)

    def test_composite_evidence_empty_by_default(self):
        pred = self._make_pred()
        assert pred.composite_evidence == []

    def test_tier_a_gate_override_default_false(self):
        """tier_a_gate_override must default to False (gate not fired)."""
        pred = self._make_pred()
        assert pred.tier_a_gate_override is False

    def test_tier_a_gate_override_true_when_set(self):
        """When gate fires (IS110 OOD probe), tier_a_gate_override must be True."""
        pred = self._make_pred(
            tier_a="DSB_FREE_TRANSEST_RECOMBINASE",
            tier_a_confidence=0.90,
            tier_a_gate_override=True,
        )
        assert pred.tier_a_gate_override is True
        assert pred.tier_a == "DSB_FREE_TRANSEST_RECOMBINASE"
        assert pred.tier_a_confidence == pytest.approx(0.90)

    def test_model_dump_returns_dict(self):
        pred = self._make_pred()
        d = pred.model_dump()
        assert isinstance(d, dict)
        assert "tier_a" in d
        assert "accession" in d
        assert "tier_a_gate_override" in d


# PFAM_WHITELIST integrity


class TestPfamWhitelist:
    def test_has_23_entries(self):
        assert len(PFAM_WHITELIST) == 23

    def test_all_entries_are_strings(self):
        assert all(isinstance(p, str) for p in PFAM_WHITELIST)

    def test_no_duplicates(self):
        assert len(PFAM_WHITELIST) == len(set(PFAM_WHITELIST))

    def test_key_pfam_accessions_present(self):
        assert "PF01548" in PFAM_WHITELIST  # IS110 N-terminal (dom_4)
        assert "PF02371" in PFAM_WHITELIST  # IS110 C-terminal (dom_5)
        assert "PF07282" in PFAM_WHITELIST  # Fanzor/TnpB (dom_6)
        assert "PF13395" in PFAM_WHITELIST  # Cas9 HNH (dom_0)

    def test_is110_domains_adjacent(self):
        """PF01548 and PF02371 must be dom_4 and dom_5 (adjacent) for IS110 composite."""
        assert PFAM_WHITELIST[4] == "PF01548"
        assert PFAM_WHITELIST[5] == "PF02371"
