"""Mocked unit tests for mech_class.api — covers predict_from_sequence,
predict_from_fasta, predict_batch, _embed_sequence, _load_esm2_singleton,
and _fetch_pfam_hits.  No trained model files required.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from mech_class.api import (
    Prediction,
    Predictor,
    _fetch_pfam_hits,
    _load_esm2_singleton,
)

# ---------------------------------------------------------------------------
# Picklable model stub (MagicMock is not picklable — used for load() tests)
# ---------------------------------------------------------------------------


class _MockLGBM:
    """Minimal picklable LightGBM stand-in for Predictor.load() tests."""

    def __init__(self, proba):
        self._proba = np.array(proba)

    def predict_proba(self, X):
        return np.tile(self._proba, (len(X), 1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASSES = ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE", "TRANSPOSASE"]
MOCK_FEAT_COLS = [f"seq_{i}" for i in range(640)] + [f"dom_{i}" for i in range(26)]

# After LabelEncoder.fit(sorted CLASSES):
#   DSB_FREE_TRANSEST_RECOMBINASE → index 0
#   DSB_NUCLEASE                  → index 1
#   TRANSPOSASE                   → index 2


def _make_label_encoder(classes=None):
    le = LabelEncoder()
    le.fit(classes or CLASSES)
    return le


def make_mock_predictor(
    ta_proba=None,
    comp_proba=None,
    tier_b_proba=None,
    include_tier_b=True,
):
    """Return a Predictor with stub LightGBM models (no model files needed)."""
    ta_proba = ta_proba if ta_proba is not None else [[0.05, 0.85, 0.10]]
    comp_proba = comp_proba if comp_proba is not None else [[0.80, 0.20]]
    tier_b_proba = tier_b_proba if tier_b_proba is not None else [[0.30, 0.70]]

    ta_le = _make_label_encoder()
    ta_model = MagicMock()
    ta_model.predict_proba.return_value = np.array(ta_proba)
    ta = {"model": ta_model, "label_encoder": ta_le, "feature_cols": MOCK_FEAT_COLS}

    comp_model = MagicMock()
    comp_model.predict_proba.return_value = np.array(comp_proba)
    comp = {"model": comp_model, "feature_cols": None}

    tier_b: dict = {}
    if include_tier_b:
        for cls in CLASSES:
            tb_le = _make_label_encoder([f"{cls}_subA", f"{cls}_subB"])
            tb_model = MagicMock()
            tb_model.predict_proba.return_value = np.array(tier_b_proba)
            tier_b[cls] = {
                "model": tb_model,
                "label_encoder": tb_le,
                "feature_cols": MOCK_FEAT_COLS,
            }

    return Predictor(ta, comp, tier_b)


# ---------------------------------------------------------------------------
# Predictor.__init__
# ---------------------------------------------------------------------------


class TestPredictorInit:
    def test_init_stores_models(self):
        ta = {"model": MagicMock(), "label_encoder": _make_label_encoder(), "feature_cols": MOCK_FEAT_COLS}
        comp = {"model": MagicMock(), "feature_cols": None}
        pred = Predictor(ta, comp, {})
        assert pred._ta is ta
        assert pred._comp is comp
        assert pred._tier_b == {}
        assert pred._esm2 is None

    def test_init_with_tier_b(self):
        pred = make_mock_predictor()
        assert set(pred._tier_b.keys()) == set(CLASSES)


# ---------------------------------------------------------------------------
# Predictor.load() — file-loading paths
# ---------------------------------------------------------------------------


class TestPredictorLoad:
    def _write_mock_models(self, tmp_path: Path) -> None:
        """Write minimal pickle files for load() to consume."""
        ta_le = _make_label_encoder()
        ta_dict = {
            "model": _MockLGBM([[0.05, 0.85, 0.10]]),
            "label_encoder": ta_le,
            "feature_cols": MOCK_FEAT_COLS,
        }
        comp_dict = {"model": _MockLGBM([[0.80, 0.20]]), "feature_cols": None}

        tb_le = _make_label_encoder(["N1_CRISPR_Cas", "N2_Fanzor"])
        tb_dict = {
            "model": _MockLGBM([[0.30, 0.70]]),
            "label_encoder": tb_le,
            "feature_cols": MOCK_FEAT_COLS,
        }

        (tmp_path / "tier_a").mkdir()
        (tmp_path / "composite_head").mkdir()
        (tmp_path / "tier_b" / "DSB_NUCLEASE").mkdir(parents=True)

        with open(tmp_path / "tier_a" / "model.pkl", "wb") as f:
            pickle.dump(ta_dict, f)
        with open(tmp_path / "composite_head" / "model.pkl", "wb") as f:
            pickle.dump(comp_dict, f)
        with open(tmp_path / "tier_b" / "DSB_NUCLEASE" / "model.pkl", "wb") as f:
            pickle.dump(tb_dict, f)

    def test_load_from_directory(self, tmp_path):
        """Predictor.load(dir) reads pickle files and returns a Predictor."""
        self._write_mock_models(tmp_path)
        pred = Predictor.load(tmp_path, download=False)
        assert isinstance(pred, Predictor)
        assert "DSB_NUCLEASE" in pred._tier_b

    def test_load_tier_b_populated(self, tmp_path):
        self._write_mock_models(tmp_path)
        pred = Predictor.load(tmp_path, download=False)
        assert len(pred._tier_b) >= 1

    def test_load_none_model_dir_download_false_raises_file_not_found(self, tmp_path):
        """model_dir=None, download=False → falls back to default cache (empty) → FileNotFoundError."""
        with (
            patch("mech_class.api._DEFAULT_CACHE_DIR", tmp_path / "empty_cache"),
            pytest.raises(FileNotFoundError, match="Tier-A model not found"),
        ):
            Predictor.load(download=False)

    def test_load_none_model_dir_triggers_zenodo_download(self, tmp_path):
        """model_dir=None, download=True → _download_from_zenodo → RuntimeError."""
        with (
            patch("mech_class.api._DEFAULT_CACHE_DIR", tmp_path / "no_cache"),
            pytest.raises(RuntimeError, match="Zenodo"),
        ):
            Predictor.load()


# ---------------------------------------------------------------------------
# predict_from_sequence
# ---------------------------------------------------------------------------


class TestPredictFromSequence:
    SEQ = "MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEIFSNEMAK"

    def test_basic_prediction_returns_prediction(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert isinstance(result, Prediction)

    def test_accession_none_uses_unknown(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert result.accession == "unknown"

    def test_accession_set(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence("Q99ZW2", self.SEQ, pfam_hits=[])
        assert result.accession == "Q99ZW2"

    def test_sequence_length_correct(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert result.sequence_length == len(self.SEQ)

    def test_tier_a_set(self):
        """Predictor returns the class with highest probability."""
        # ta_proba [0.05, 0.85, 0.10] → argmax=1 → DSB_NUCLEASE
        pred = make_mock_predictor(ta_proba=[[0.05, 0.85, 0.10]])
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert result.tier_a == "DSB_NUCLEASE"

    def test_tier_a_confidence_set(self):
        pred = make_mock_predictor(ta_proba=[[0.05, 0.85, 0.10]])
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert result.tier_a_confidence == pytest.approx(0.85)

    def test_no_gate_override_when_no_is110_domains(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=["PF13395"])
        assert result.tier_a_gate_override is False

    def test_is110_gate_overrides_when_both_domains_present(self):
        """PF01548 + PF02371 present, model predicts DSB_NUCLEASE → gate fires."""
        pred = make_mock_predictor(ta_proba=[[0.05, 0.85, 0.10]])
        result = pred.predict_from_sequence(
            None,
            self.SEQ,
            pfam_hits=["PF01548", "PF02371"],
        )
        assert result.tier_a == "DSB_FREE_TRANSEST_RECOMBINASE"
        assert result.tier_a_gate_override is True
        assert result.tier_a_confidence >= 0.90

    def test_is110_gate_does_not_fire_when_already_correct(self):
        """If ML already predicts DSB_FREE, gate does not mark override."""
        # ta_proba [0.85, 0.05, 0.10] → argmax=0 → DSB_FREE_TRANSEST_RECOMBINASE
        pred = make_mock_predictor(ta_proba=[[0.85, 0.05, 0.10]])
        result = pred.predict_from_sequence(
            None,
            self.SEQ,
            pfam_hits=["PF01548", "PF02371"],
        )
        assert result.tier_a == "DSB_FREE_TRANSEST_RECOMBINASE"
        assert result.tier_a_gate_override is False

    def test_composite_true_when_gate_passes_and_ml_high(self):
        """PF01548 + PF02371 + composite ML prob ≥ 0.5 → composite=True."""
        pred = make_mock_predictor(
            ta_proba=[[0.85, 0.05, 0.10]],
            comp_proba=[[0.10, 0.90]],  # index 1 = composite prob 0.90
        )
        result = pred.predict_from_sequence(
            None,
            self.SEQ,
            pfam_hits=["PF01548", "PF02371"],
        )
        assert result.composite is True
        assert result.composite_prob == pytest.approx(0.90)
        assert len(result.composite_evidence) == 2  # two domain strings

    def test_composite_false_when_gate_fails(self):
        """No IS110 domains → gate does not pass → composite always False."""
        pred = make_mock_predictor(comp_proba=[[0.10, 0.90]])
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=["PF13395"])
        assert result.composite is False
        assert result.composite_prob == pytest.approx(0.0)

    def test_composite_evidence_low_ml_when_gate_passes(self):
        """Gate passes (domains present) but ML below 0.5 → low-confidence evidence string."""
        pred = make_mock_predictor(
            ta_proba=[[0.85, 0.05, 0.10]],
            comp_proba=[[0.70, 0.30]],  # prob=0.30 < 0.5 → not composite
        )
        result = pred.predict_from_sequence(
            None,
            self.SEQ,
            pfam_hits=["PF01548", "PF02371"],
        )
        assert result.composite is False
        # Should contain a low-confidence evidence note
        assert len(result.composite_evidence) == 1
        assert "low" in result.composite_evidence[0].lower()

    def test_tier_b_populated(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(
            None,
            self.SEQ,
            pfam_hits=["PF13395"],
        )
        assert result.tier_b is not None
        assert result.tier_b_confidence is not None

    def test_tier_b_none_when_not_in_tier_b_dict(self):
        """If no Tier-B model for the predicted class, tier_b is None."""
        pred = make_mock_predictor(include_tier_b=False)
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert result.tier_b is None
        assert result.tier_b_confidence is None

    def test_f_domain_channel_added_when_pfam_hits_present(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=["PF13395"])
        assert "F_domain" in result.channels_used

    def test_no_f_domain_channel_when_pfam_empty(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert "F_domain" not in result.channels_used

    def test_pfam_hits_stored_in_result(self):
        pred = make_mock_predictor()
        pfam = ["PF01548", "PF02371"]
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=pfam)
        assert result.pfam_hits == pfam

    def test_pfam_none_triggers_uniprot_lookup(self):
        """pfam_hits=None with accession calls _fetch_pfam_hits (mocked)."""
        pred = make_mock_predictor()
        with patch("mech_class.api._fetch_pfam_hits", return_value=["PF13395"]) as mock_fetch:
            result = pred.predict_from_sequence("Q99ZW2", self.SEQ)
        mock_fetch.assert_called_once_with("Q99ZW2")
        assert "PF13395" in result.pfam_hits

    def test_pfam_none_no_accession_gives_empty_pfam(self):
        """pfam_hits=None + accession=None → pfam_hits = [] (no network call)."""
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ)
        assert result.pfam_hits == []

    def test_f_seq_channel_added_when_embed_sequence_returns_array(self):
        pred = make_mock_predictor()
        fake_emb = np.ones(640, dtype=np.float32)
        with patch.object(pred, "_embed_sequence", return_value=fake_emb):
            result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert "F_seq" in result.channels_used

    def test_f_seq_channel_absent_when_embed_returns_none(self):
        pred = make_mock_predictor()
        with patch.object(pred, "_embed_sequence", return_value=None):
            result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        assert "F_seq" not in result.channels_used

    def test_prediction_is_serialisable_to_dict(self):
        pred = make_mock_predictor()
        result = pred.predict_from_sequence(None, self.SEQ, pfam_hits=[])
        d = result.model_dump()
        assert "tier_a" in d
        assert "composite" in d


# ---------------------------------------------------------------------------
# predict_from_fasta
# ---------------------------------------------------------------------------


class TestPredictFromFasta:
    SEQ = "MDKKYSIGLDIG"

    def test_predict_from_fasta_returns_list(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(f">Q99ZW2\n{self.SEQ}\n")

        pred = make_mock_predictor()
        results = pred.predict_from_fasta(str(fasta))
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].accession == "Q99ZW2"

    def test_predict_from_fasta_multi_sequence(self, tmp_path):
        fasta = tmp_path / "multi.fasta"
        fasta.write_text(f">ACC1\n{self.SEQ}\n>ACC2\n{self.SEQ}\n")

        pred = make_mock_predictor()
        results = pred.predict_from_fasta(str(fasta))
        assert len(results) == 2

    def test_predict_from_fasta_empty_file(self, tmp_path):
        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")

        pred = make_mock_predictor()
        results = pred.predict_from_fasta(str(fasta))
        assert results == []

    def test_predict_from_fasta_no_biopython_raises(self):
        """If biopython is not installed, ImportError is raised with guidance."""
        pred = make_mock_predictor()
        # Simulate missing biopython by blocking the import
        with patch.dict(sys.modules, {"Bio": None, "Bio.SeqIO": None}), pytest.raises(ImportError, match="biopython"):
            pred.predict_from_fasta("dummy.fasta")


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    SEQ = "MDKKYSIGLDIG"

    def _make_df(self, with_pfam=True, pfam_nan=False):
        data = {
            "accession": ["ACC1", "ACC2"],
            "sequence": [self.SEQ, self.SEQ],
        }
        if with_pfam:
            data["pfam_hits"] = [["PF13395"], float("nan") if pfam_nan else ["PF01548"]]
        return pd.DataFrame(data)

    def test_predict_batch_returns_dataframe(self):
        pred = make_mock_predictor()
        df = self._make_df()
        result = pred.predict_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_predict_batch_columns_present(self):
        pred = make_mock_predictor()
        df = self._make_df()
        result = pred.predict_batch(df)
        assert "tier_a" in result.columns
        assert "accession" in result.columns

    def test_predict_batch_nan_pfam_treated_as_none(self):
        """NaN in pfam_hits column is treated as None (no network error)."""
        pred = make_mock_predictor()
        df = self._make_df(pfam_nan=True)
        # Should not raise — NaN → None → pfam_hits=[] (no accession lookup)
        result = pred.predict_batch(df)
        assert len(result) == 2

    def test_predict_batch_no_pfam_col(self):
        """DataFrame without pfam_hits column — pfam_col defaults don't raise."""
        pred = make_mock_predictor()
        df = self._make_df(with_pfam=False)
        result = pred.predict_batch(df)
        assert len(result) == 2

    def test_predict_batch_pfam_col_none(self):
        """pfam_col=None forces UniProt lookup (mocked here)."""
        pred = make_mock_predictor()
        df = self._make_df(with_pfam=False)
        with patch("mech_class.api._fetch_pfam_hits", return_value=[]):
            result = pred.predict_batch(df, pfam_col=None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _embed_sequence
# ---------------------------------------------------------------------------


class TestEmbedSequence:
    SEQ = "MDKKYSIGLDIG"

    def test_returns_none_when_esm2_singleton_unavailable(self):
        """_esm2=None + singleton returns None → _embed_sequence returns None."""
        pred = make_mock_predictor()
        pred._esm2 = None

        import mech_class.api as api_mod

        original_singleton = api_mod._ESM2_SINGLETON
        api_mod._ESM2_SINGLETON = None
        try:
            with patch("mech_class.api._load_esm2_singleton", return_value=None):
                result = pred._embed_sequence(self.SEQ)
        finally:
            api_mod._ESM2_SINGLETON = original_singleton

        assert result is None

    def test_returns_none_when_esm2_already_set_none_after_load(self):
        """Verifies the second None check (self._esm2 remains None) returns None."""
        pred = make_mock_predictor()
        pred._esm2 = None
        with patch("mech_class.api._load_esm2_singleton", return_value=None):
            result = pred._embed_sequence(self.SEQ)
        assert result is None

    def test_embed_sequence_exception_returns_none_with_warning(self):
        """If batch_converter raises, _embed_sequence returns None and warns."""
        pred = make_mock_predictor()
        mock_converter = MagicMock(side_effect=RuntimeError("test error"))
        pred._esm2 = (MagicMock(), MagicMock(), mock_converter)

        mock_torch = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.no_grad.return_value = mock_ctx

        with patch.dict(sys.modules, {"torch": mock_torch}), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pred._embed_sequence(self.SEQ)

        assert result is None
        assert any("ESM-2 embedding failed" in str(x.message) for x in w)

    def test_embed_sequence_uses_singleton_on_first_call(self):
        """_esm2=None triggers _load_esm2_singleton call."""
        pred = make_mock_predictor()
        pred._esm2 = None
        mock_singleton = (MagicMock(), MagicMock(), MagicMock(side_effect=RuntimeError("err")))

        mock_torch = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.no_grad.return_value = mock_ctx

        with (
            patch("mech_class.api._load_esm2_singleton", return_value=mock_singleton),
            patch.dict(sys.modules, {"torch": mock_torch}),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            result = pred._embed_sequence(self.SEQ)

        assert pred._esm2 is mock_singleton
        assert result is None  # RuntimeError from converter → exception path


# ---------------------------------------------------------------------------
# _load_esm2_singleton
# ---------------------------------------------------------------------------


class TestLoadESM2Singleton:
    def _reset_singleton(self):
        import mech_class.api as api_mod

        original = api_mod._ESM2_SINGLETON
        api_mod._ESM2_SINGLETON = None
        return original

    def _restore_singleton(self, original):
        import mech_class.api as api_mod

        api_mod._ESM2_SINGLETON = original

    def test_returns_none_when_esm_not_installed(self):
        """Without fair-esm installed, function returns None (graceful fallback)."""
        original = self._reset_singleton()
        try:
            # Ensure esm is not importable
            with patch.dict(sys.modules, {"esm": None}):
                result = _load_esm2_singleton()
        finally:
            self._restore_singleton(original)
        assert result is None

    def test_returns_cached_singleton_on_second_call(self):
        """If singleton already loaded, returns it without re-importing."""
        import mech_class.api as api_mod

        original = api_mod._ESM2_SINGLETON
        fake_singleton = (MagicMock(), MagicMock(), MagicMock())
        api_mod._ESM2_SINGLETON = fake_singleton
        try:
            result = _load_esm2_singleton()
        finally:
            api_mod._ESM2_SINGLETON = original
        assert result is fake_singleton

    def test_loads_and_caches_when_esm_available(self):
        """With mocked fair-esm, function loads the model and caches the singleton."""
        original = self._reset_singleton()
        try:
            mock_esm = MagicMock()
            mock_model = MagicMock()
            mock_alphabet = MagicMock()
            mock_esm.pretrained.esm2_t30_150M_UR50D.return_value = (mock_model, mock_alphabet)
            mock_model.eval.return_value = mock_model

            with patch.dict(sys.modules, {"esm": mock_esm}):
                result = _load_esm2_singleton()
        finally:
            self._restore_singleton(original)

        assert result is not None
        assert len(result) == 3  # (model, alphabet, batch_converter)

    def test_returns_none_on_unexpected_exception(self):
        """If esm raises unexpectedly, returns None without raising."""
        original = self._reset_singleton()
        try:
            mock_esm = MagicMock()
            mock_esm.pretrained.esm2_t30_150M_UR50D.side_effect = RuntimeError("CUDA OOM")
            with patch.dict(sys.modules, {"esm": mock_esm}):
                result = _load_esm2_singleton()
        finally:
            self._restore_singleton(original)
        assert result is None


# ---------------------------------------------------------------------------
# _fetch_pfam_hits — success path
# ---------------------------------------------------------------------------


class TestFetchPfamHitsSuccess:
    def _mock_uniprot_response(self, pfam_ids):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "uniProtKBCrossReferences": [{"database": "Pfam", "id": pid} for pid in pfam_ids]
            + [{"database": "Gene3D", "id": "1.10.10.10"}]
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_pfam_ids_from_uniprot(self):
        with patch("mech_class.api.requests.get", return_value=self._mock_uniprot_response(["PF13395", "PF18541"])):
            result = _fetch_pfam_hits("Q99ZW2")
        assert result == ["PF13395", "PF18541"]

    def test_filters_non_pfam_references(self):
        with patch("mech_class.api.requests.get", return_value=self._mock_uniprot_response(["PF13395"])):
            result = _fetch_pfam_hits("Q99ZW2")
        assert "Gene3D" not in str(result)
        assert result == ["PF13395"]

    def test_returns_empty_when_no_pfam_xrefs(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"uniProtKBCrossReferences": [{"database": "Gene3D", "id": "x"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("mech_class.api.requests.get", return_value=mock_resp):
            result = _fetch_pfam_hits("Q99ZW2")
        assert result == []

    def test_returns_empty_on_404(self):
        import requests as req_lib

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError("404")
        with patch("mech_class.api.requests.get", return_value=mock_resp):
            result = _fetch_pfam_hits("INVALID")
        assert result == []
