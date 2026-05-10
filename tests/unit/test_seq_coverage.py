"""Coverage tests for mech_class.features.seq - paths not covered by
test_seq_features.py.  Covers:
  - load_esm2_singleton early-return path (singleton already loaded)
  - load_esm2_singleton success path (with mocked fair-esm)
  - load_esm2_singleton ImportError path
  - load_esm2_singleton generic Exception path
  - embed_sequence when model IS loaded (mock torch block)
  - load_esm2_embeddings validation error (missing column)
  - get_esm2_vector with df provided (found + KeyError paths)
  - build_seq_feature_matrix with esm2_df=None (triggers load)
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import mech_class.features.seq as seq_mod
from mech_class.features.seq import (
    ESM2_DIM,
    build_seq_feature_matrix,
    embed_sequence,
    get_esm2_vector,
    load_esm2_embeddings,
    load_esm2_singleton,
)

SHORT_SEQ = "MDKKYSIGLDIGTNSVGW"


# ---------------------------------------------------------------------------
# Helpers - save/restore module globals so tests don't bleed into each other
# ---------------------------------------------------------------------------


def _clear_esm2_globals():
    original = (seq_mod._ESM2_MODEL, seq_mod._ESM2_ALPHABET, seq_mod._ESM2_CONVERTER)
    seq_mod._ESM2_MODEL = None
    seq_mod._ESM2_ALPHABET = None
    seq_mod._ESM2_CONVERTER = None
    return original


def _restore_esm2_globals(original):
    seq_mod._ESM2_MODEL, seq_mod._ESM2_ALPHABET, seq_mod._ESM2_CONVERTER = original


# ---------------------------------------------------------------------------
# load_esm2_singleton - uncovered branches
# ---------------------------------------------------------------------------


class TestLoadESM2SingletonCoverage:
    def test_early_return_true_when_already_loaded(self):
        """Line 35: if _ESM2_MODEL is not None -> return True immediately."""
        original = _clear_esm2_globals()
        seq_mod._ESM2_MODEL = MagicMock()  # pretend already loaded
        try:
            result = load_esm2_singleton(verbose=False)
        finally:
            _restore_esm2_globals(original)
        assert result is True

    def test_success_with_mock_esm(self):
        """Lines 39-47: successful ESM-2 load with mocked fair-esm."""
        original = _clear_esm2_globals()
        mock_esm = MagicMock()
        mock_model = MagicMock()
        mock_alphabet = MagicMock()
        mock_esm.pretrained.esm2_t30_150M_UR50D.return_value = (mock_model, mock_alphabet)
        mock_model.eval.return_value = mock_model

        try:
            with patch.dict(sys.modules, {"esm": mock_esm}):
                result = load_esm2_singleton(verbose=False)
        finally:
            _restore_esm2_globals(original)

        assert result is True

    def test_success_with_verbose_print(self, capsys):
        """Lines 45-46: verbose=True prints a status message."""
        original = _clear_esm2_globals()
        mock_esm = MagicMock()
        mock_model = MagicMock()
        mock_alphabet = MagicMock()
        mock_esm.pretrained.esm2_t30_150M_UR50D.return_value = (mock_model, mock_alphabet)
        mock_model.eval.return_value = mock_model

        try:
            with patch.dict(sys.modules, {"esm": mock_esm}):
                result = load_esm2_singleton(verbose=True)
        finally:
            _restore_esm2_globals(original)

        captured = capsys.readouterr()
        assert "ESM-2" in captured.out
        assert result is True

    def test_import_error_returns_false_with_warning(self):
        """Lines 48-49: ImportError -> warn and return False."""
        original = _clear_esm2_globals()
        try:
            with patch.dict(sys.modules, {"esm": None}), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = load_esm2_singleton(verbose=False)
        finally:
            _restore_esm2_globals(original)

        assert result is False
        assert any("fair-esm" in str(x.message).lower() or "esm" in str(x.message).lower() for x in w)

    def test_generic_exception_returns_false_with_warning(self):
        """Lines 50-52: non-ImportError exception -> warn and return False."""
        original = _clear_esm2_globals()
        mock_esm = MagicMock()
        mock_esm.pretrained.esm2_t30_150M_UR50D.side_effect = RuntimeError("CUDA out of memory")
        try:
            with patch.dict(sys.modules, {"esm": mock_esm}), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = load_esm2_singleton(verbose=False)
        finally:
            _restore_esm2_globals(original)

        assert result is False
        assert any("ESM-2 load failed" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# embed_sequence - torch computation block (lines 79-98)
# ---------------------------------------------------------------------------


class TestEmbedSequenceWithModel:
    def test_embed_sequence_exception_returns_zero_vector(self):
        """Lines 96-98: exception in torch block -> return zero vector with warning."""
        original = _clear_esm2_globals()
        # Set a mock model that will trigger the torch path
        seq_mod._ESM2_MODEL = MagicMock()
        # Converter raises -> caught by except Exception
        seq_mod._ESM2_CONVERTER = MagicMock(side_effect=RuntimeError("inference error"))

        mock_torch = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.no_grad.return_value = mock_ctx

        try:
            with patch.dict(sys.modules, {"torch": mock_torch}), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = embed_sequence(SHORT_SEQ)
        finally:
            _restore_esm2_globals(original)

        assert result.shape == (ESM2_DIM,)
        assert result.dtype == np.float32
        assert any("ESM-2 inference failed" in str(x.message) for x in w)

    def test_embed_sequence_torch_computation(self):
        """Lines 80-94: full torch computation path with mocked torch."""
        original = _clear_esm2_globals()

        # Set up mock model globals
        expected_emb = np.ones(ESM2_DIM, dtype=np.float32) * 0.42

        # Build mock return chain: model output -> representations -> slice -> mean -> cpu -> numpy -> astype
        mock_mean = MagicMock()
        mock_mean.cpu.return_value.numpy.return_value.astype.return_value = expected_emb
        mock_slice = MagicMock()
        mock_slice.mean.return_value = mock_mean
        mock_repr = MagicMock()
        mock_repr.__getitem__.return_value = mock_slice
        mock_out = {"representations": {30: mock_repr}}

        mock_model = MagicMock(return_value=mock_out)
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_converter = MagicMock(return_value=(None, None, mock_tokens))

        seq_mod._ESM2_MODEL = mock_model
        seq_mod._ESM2_CONVERTER = mock_converter

        mock_torch = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.no_grad.return_value = mock_ctx

        try:
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = embed_sequence(SHORT_SEQ)
        finally:
            _restore_esm2_globals(original)

        np.testing.assert_array_equal(result, expected_emb)

    def test_embed_sequence_gpu_device_path(self):
        """Lines 84-85: device!='cpu' -> model.to(device) is called."""
        original = _clear_esm2_globals()

        expected_emb = np.zeros(ESM2_DIM, dtype=np.float32)
        mock_mean = MagicMock()
        mock_mean.cpu.return_value.numpy.return_value.astype.return_value = expected_emb
        mock_slice = MagicMock()
        mock_slice.mean.return_value = mock_mean
        mock_repr = MagicMock()
        mock_repr.__getitem__.return_value = mock_slice
        mock_out = {"representations": {30: mock_repr}}

        mock_model = MagicMock(return_value=mock_out)
        mock_model.to.return_value = mock_model  # model.to("cuda") returns itself
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_converter = MagicMock(return_value=(None, None, mock_tokens))

        seq_mod._ESM2_MODEL = mock_model
        seq_mod._ESM2_CONVERTER = mock_converter

        mock_torch = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.no_grad.return_value = mock_ctx

        try:
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = embed_sequence(SHORT_SEQ, device="cuda")
        finally:
            _restore_esm2_globals(original)

        # model.to("cuda") should have been called
        mock_model.to.assert_called_once_with("cuda")
        assert result.shape == (ESM2_DIM,)


# ---------------------------------------------------------------------------
# load_esm2_embeddings - validation error path
# ---------------------------------------------------------------------------


class TestLoadESM2EmbeddingsValidation:
    def test_missing_accession_column_raises(self, tmp_path):
        """Lines 111-113: missing 'accession' or 'embedding' column -> ValueError."""
        bad_parquet = tmp_path / "bad.parquet"
        pd.DataFrame({"wrong_col": [1, 2]}).to_parquet(bad_parquet)
        with pytest.raises(ValueError, match="'accession' or 'embedding'"):
            load_esm2_embeddings(bad_parquet)

    def test_missing_embedding_column_raises(self, tmp_path):
        bad_parquet = tmp_path / "bad2.parquet"
        pd.DataFrame({"accession": ["A1"], "other": [1]}).to_parquet(bad_parquet)
        with pytest.raises(ValueError, match="'accession' or 'embedding'"):
            load_esm2_embeddings(bad_parquet)

    def test_valid_parquet_returns_dataframe(self, tmp_path):
        good_parquet = tmp_path / "good.parquet"
        pd.DataFrame(
            {
                "accession": ["ACC1"],
                "embedding": [np.zeros(ESM2_DIM, dtype=np.float32).tolist()],
            }
        ).to_parquet(good_parquet)
        df = load_esm2_embeddings(good_parquet)
        assert isinstance(df, pd.DataFrame)
        assert "accession" in df.columns


# ---------------------------------------------------------------------------
# get_esm2_vector - found + KeyError paths
# ---------------------------------------------------------------------------


class TestGetESM2Vector:
    def _make_df(self, accessions: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "accession": accessions,
                "embedding": [np.ones(ESM2_DIM, dtype=np.float32).tolist() for _ in accessions],
            }
        )

    def test_returns_vector_when_found(self):
        """Lines 137, 140: accession found in df -> return vector."""
        df = self._make_df(["ACC1", "ACC2"])
        vec = get_esm2_vector("ACC1", df=df)
        assert vec.shape == (ESM2_DIM,)
        assert vec.dtype == np.float32
        assert np.all(vec == 1.0)

    def test_raises_key_error_when_not_found(self):
        """Lines 138-139: accession not in df -> KeyError."""
        df = self._make_df(["ACC1"])
        with pytest.raises(KeyError, match="MISSING"):
            get_esm2_vector("MISSING", df=df)

    def test_loads_from_parquet_when_df_none(self, tmp_path):
        """Line 136: df=None -> calls load_esm2_embeddings(default path)."""
        good_parquet = tmp_path / "emb.parquet"
        pd.DataFrame(
            {
                "accession": ["ACC_X"],
                "embedding": [np.ones(ESM2_DIM, dtype=np.float32).tolist()],
            }
        ).to_parquet(good_parquet)
        with patch(
            "mech_class.features.seq.load_esm2_embeddings",
            return_value=pd.DataFrame(
                {
                    "accession": ["ACC_X"],
                    "embedding": [np.ones(ESM2_DIM, dtype=np.float32).tolist()],
                }
            ),
        ):
            vec = get_esm2_vector("ACC_X", df=None)
        assert vec.shape == (ESM2_DIM,)


# ---------------------------------------------------------------------------
# build_seq_feature_matrix - esm2_df=None path (line 169)
# ---------------------------------------------------------------------------


class TestBuildSeqFeatureMatrixNullDf:
    def test_none_df_calls_load_esm2_embeddings(self):
        """Line 169: esm2_df=None -> load_esm2_embeddings() is called."""
        fake_df = pd.DataFrame(
            {
                "accession": ["ACC_A"],
                "embedding": [np.ones(ESM2_DIM, dtype=np.float32).tolist()],
            }
        )
        with patch("mech_class.features.seq.load_esm2_embeddings", return_value=fake_df) as mock_load:
            matrix = build_seq_feature_matrix(["ACC_A"], esm2_df=None)
        mock_load.assert_called_once()
        assert matrix.shape == (1, ESM2_DIM)

    def test_none_df_missing_accession_still_returns_zeros(self):
        """When esm2_df loaded but accession missing -> zero row + warning."""
        fake_df = pd.DataFrame(
            {
                "accession": ["OTHER"],
                "embedding": [np.ones(ESM2_DIM, dtype=np.float32).tolist()],
            }
        )
        with (
            patch("mech_class.features.seq.load_esm2_embeddings", return_value=fake_df),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            matrix = build_seq_feature_matrix(["MISSING"], esm2_df=None)
        assert matrix.shape == (1, ESM2_DIM)
        assert np.all(matrix[0] == 0.0)
        assert any("missing" in str(x.message).lower() for x in w)
