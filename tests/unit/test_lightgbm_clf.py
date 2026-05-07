"""Unit tests for LightGBMClassifier — fit, predict, bootstrap CI.

Uses the actual class name LightGBMClassifier from mech_class.models.lightgbm_clf.
Skipped automatically if lightgbm is not installed (local dev without full deps).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed; install with: pip install mech-class[dev]")

from mech_class.models.lightgbm_clf import LightGBMClassifier

RNG     = np.random.default_rng(42)
CLASSES = ["DSB_NUCLEASE", "DSB_FREE_TRANSEST_RECOMBINASE", "TRANSPOSASE"]


def _make_data(n_per_class: int = 20, n_features: int = 30):
    X_parts, y_parts = [], []
    for i, cls in enumerate(CLASSES):
        X = RNG.normal(loc=i * 2.0, scale=1.0, size=(n_per_class, n_features)).astype(np.float32)
        X_parts.append(X)
        y_parts.extend([cls] * n_per_class)
    return np.vstack(X_parts), np.array(y_parts)


class TestLightGBMClassifier:
    def setup_method(self):
        self.clf = LightGBMClassifier(classes=CLASSES, random_state=42)
        self.X, self.y = _make_data()

    def test_fit_predict(self):
        self.clf.fit(self.X, self.y)
        preds = self.clf.predict(self.X)
        assert len(preds) == len(self.y)
        assert set(preds).issubset(set(CLASSES))

    def test_predict_proba_shape(self):
        self.clf.fit(self.X, self.y)
        proba = self.clf.predict_proba(self.X)
        assert proba.shape == (len(self.y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_macro_f1_returns_triple(self):
        self.clf.fit(self.X, self.y)
        point, lo, hi = self.clf.macro_f1(self.X, self.y, n_bootstrap=50, seed=42)
        assert 0.0 <= lo <= point <= hi <= 1.0

    def test_feature_importances(self):
        self.clf.fit(self.X, self.y)
        fi = self.clf.feature_importances()
        assert len(fi) == self.X.shape[1]

    def test_save_load(self, tmp_path):
        self.clf.fit(self.X, self.y)
        path = tmp_path / "model.lgb"
        self.clf.save(path)
        clf2 = LightGBMClassifier.load(path)
        preds1 = self.clf.predict(self.X)
        preds2 = clf2.predict(self.X)
        assert (preds1 == preds2).all()

    def test_predict_proba_top(self):
        self.clf.fit(self.X, self.y)
        one_sample = self.X[0]
        label, conf = self.clf.predict_proba_top(one_sample)
        assert label in CLASSES
        assert 0.0 <= conf <= 1.0

    def test_separable_data_achieves_high_f1(self):
        self.clf.fit(self.X, self.y)
        point, lo, hi = self.clf.macro_f1(self.X, self.y, n_bootstrap=50, seed=42)
        assert point >= 0.70, f"Expected macro-F1 ≥ 0.70 on separable data, got {point:.3f}"
