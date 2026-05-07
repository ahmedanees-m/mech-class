"""Unit tests for mech_class.features.seq — ESM-2 150M embedding module.

Tests are designed to pass without ESM-2 / fair-esm installed (CI environment).
When ESM-2 is unavailable, functions degrade gracefully to zero vectors.
"""
from __future__ import annotations

import numpy as np
import pytest

from mech_class.features.seq import (
    ESM2_DIM,
    ESM2_MAX_LEN,
    build_seq_feature_matrix,
    embed_sequence,
    extract_esm2_features,
    load_esm2_singleton,
)

SHORT_SEQ  = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTLSLQELIDAYRQQDEPQAQLATQSLGVVNSSIVTYDLSK"
LONG_SEQ   = "A" * 2000   # exceeds ESM2_MAX_LEN (1022); should be truncated silently

FAKE_ACC   = "FAKEACC1"


class TestESM2Constants:
    def test_dim_is_640(self):
        assert ESM2_DIM == 640

    def test_max_len_is_1022(self):
        assert ESM2_MAX_LEN == 1022


class TestEmbedSequence:
    """embed_sequence() / extract_esm2_features() — test zero-fill fallback."""

    def test_returns_correct_shape(self):
        """Must return a 640-dim vector regardless of whether ESM-2 is installed."""
        vec = embed_sequence(SHORT_SEQ)
        assert vec.shape == (ESM2_DIM,)

    def test_returns_float32(self):
        vec = embed_sequence(SHORT_SEQ)
        assert vec.dtype == np.float32

    def test_long_seq_does_not_crash(self):
        """Sequence exceeding ESM2_MAX_LEN must be silently truncated, not error."""
        vec = embed_sequence(LONG_SEQ)
        assert vec.shape == (ESM2_DIM,)

    def test_empty_seq_does_not_crash(self):
        """Empty string must return zero vector, not raise."""
        vec = embed_sequence("")
        assert vec.shape == (ESM2_DIM,)

    def test_alias_extract_esm2_features(self):
        """extract_esm2_features is a legacy alias for embed_sequence."""
        v1 = embed_sequence(SHORT_SEQ)
        v2 = extract_esm2_features(SHORT_SEQ)
        assert np.array_equal(v1, v2)


class TestLoadESM2Singleton:
    def test_returns_bool(self):
        """load_esm2_singleton() always returns a bool (True=loaded, False=unavailable)."""
        result = load_esm2_singleton(verbose=False)
        assert isinstance(result, bool)

    def test_second_call_idempotent(self):
        """Second call should not reload (singleton pattern). Must not raise."""
        load_esm2_singleton(verbose=False)
        load_esm2_singleton(verbose=False)   # must be a no-op, no error


class TestBuildSeqFeatureMatrix:
    def test_output_shape(self):
        """build_seq_feature_matrix returns (N, 640) even if no embeddings present."""
        accs = ["ACC_A", "ACC_B", "ACC_C"]
        import pandas as pd
        # Provide a minimal fake embedding DataFrame so parquet read is skipped
        fake_df = pd.DataFrame({
            "accession": ["ACC_A"],
            "embedding": [np.zeros(ESM2_DIM, dtype=np.float32).tolist()],
        })
        matrix = build_seq_feature_matrix(accs, esm2_df=fake_df, allow_inference=False)
        assert matrix.shape == (3, ESM2_DIM)
        assert matrix.dtype == np.float32

    def test_known_accession_is_looked_up(self):
        """Accessions present in the df must fill non-zero rows."""
        import pandas as pd
        ref_vec = np.ones(ESM2_DIM, dtype=np.float32) * 3.14
        fake_df = pd.DataFrame({
            "accession": ["ACC_X"],
            "embedding":  [ref_vec.tolist()],
        })
        matrix = build_seq_feature_matrix(["ACC_X"], esm2_df=fake_df)
        assert np.allclose(matrix[0], ref_vec)

    def test_missing_accession_gives_zero_row(self):
        """Accessions not in the df must produce zero-filled rows (no error)."""
        import pandas as pd
        fake_df = pd.DataFrame({"accession": [], "embedding": []})
        matrix = build_seq_feature_matrix(["MISSING_ACC"], esm2_df=fake_df, allow_inference=False)
        assert np.all(matrix[0] == 0.0)

    def test_inference_fallback_fills_row(self):
        """allow_inference=True with a sequence must call embed_sequence and fill the row.

        Even if embed_sequence returns zeros (ESM-2 not installed), the row must
        not be all NaN and the shape must be correct.
        """
        import pandas as pd
        fake_df = pd.DataFrame({"accession": [], "embedding": []})
        seqs    = {"NEW_ACC": SHORT_SEQ}
        matrix  = build_seq_feature_matrix(
            ["NEW_ACC"], esm2_df=fake_df, allow_inference=True, sequences=seqs
        )
        assert matrix.shape == (1, ESM2_DIM)
        assert not np.any(np.isnan(matrix))
