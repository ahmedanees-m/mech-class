"""Unit tests for CompositeHead - binary IS110 classifier with FP guard.

Skipped automatically if lightgbm is not installed (local dev without full deps).

Uses the actual API:
  CompositeHead(seed=42)           - constructor parameter is `seed` not `random_seed`
  head.predict_batch(X)            - array-in / array-out; predict(x) returns (bool, list)
  head.predict_proba_batch(X)      - returns probability of composite=True
  head.fp_rate(X, y)               - false positive rate on negatives
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed; install with: pip install mech-class[dev]")

from mech_class.models.composite_head import CompositeHead

RNG = np.random.default_rng(0)


def _make_binary_data(n_pos: int = 15, n_neg: int = 45, n_features: int = 20):
    X_pos = RNG.normal(loc=3.0, scale=0.8, size=(n_pos, n_features)).astype(np.float32)
    X_neg = RNG.normal(loc=0.0, scale=1.0, size=(n_neg, n_features)).astype(np.float32)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    return X, y


class TestCompositeHead:
    def setup_method(self):
        self.head = CompositeHead(seed=42)  # parameter is `seed`, not `random_seed`
        self.X, self.y = _make_binary_data()

    def test_fit_predict_batch(self):
        self.head.fit(self.X, self.y)
        preds = self.head.predict_batch(self.X)
        assert len(preds) == len(self.y)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_batch_shape(self):
        self.head.fit(self.X, self.y)
        proba = self.head.predict_proba_batch(self.X)
        assert proba.shape == (len(self.y),)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_predict_single_returns_tuple(self):
        """CompositeHead.predict(x) -> (bool, list[str]) for one sample."""
        self.head.fit(self.X, self.y)
        composite, evidence = self.head.predict(self.X[0])
        assert isinstance(composite, bool)
        assert isinstance(evidence, list)

    def test_fp_rate_acceptable_on_separable(self):
        self.head.fit(self.X, self.y)
        fp_rate = self.head.fp_rate(self.X, self.y)
        assert fp_rate <= 0.20, f"FP rate {fp_rate:.2f} too high on separable test data"

    def test_set_threshold_changes_predictions(self):
        self.head.fit(self.X, self.y)
        preds_default = self.head.predict_batch(self.X).copy()
        self.head.set_threshold(0.99)  # very conservative -> fewer positives
        preds_strict = self.head.predict_batch(self.X)
        # Strict threshold should predict fewer (or equal) composites
        assert preds_strict.sum() <= preds_default.sum()
        # Reset threshold
        self.head.set_threshold(0.5)

    def test_save_load(self, tmp_path):
        self.head.fit(self.X, self.y)
        path = tmp_path / "composite.lgb"
        self.head.save(path)
        head2 = CompositeHead.load(path)
        preds1 = self.head.predict_batch(self.X)
        preds2 = head2.predict_batch(self.X)
        assert (preds1 == preds2).all()
