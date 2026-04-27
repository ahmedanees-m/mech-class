"""LightGBM multi-class classifier wrapper for mech-class.

Wraps ``lgb.LGBMClassifier`` with:
 - sklearn-compatible fit / predict / predict_proba
 - Bootstrap confidence intervals for macro-F1
 - save / load as plain pickle
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGBM_AVAILABLE = False


class LightGBMClassifier:
    """Multi-class LightGBM wrapper.

    Parameters
    ----------
    classes : sequence of str
        Ordered class names.  Predictions are decoded using this list.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of boosting rounds.
    **lgb_kwargs
        Extra kwargs forwarded to ``lgb.LGBMClassifier``.
    """

    def __init__(
        self,
        classes: Sequence[str],
        *,
        random_state: int = 42,
        n_estimators: int = 200,
        **lgb_kwargs,
    ):
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm is required: pip install mech-class[dev]")
        self.classes_ = list(classes)
        self.random_state = random_state
        self._model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(self.classes_),
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1,
            **lgb_kwargs,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBMClassifier:
        """Fit the classifier.

        Parameters
        ----------
        X : (n, d) float32 array
        y : (n,) string array of class labels
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix, shape (n_samples, n_classes)."""
        proba = self._model.predict_proba(X)
        if proba.ndim == 1:
            proba = proba.reshape(-1, len(self.classes_))
        return proba.astype(np.float32)

    def predict_proba_top(self, x: np.ndarray) -> tuple[str, float]:
        """Return (top_class_label, confidence) for a single sample.

        Parameters
        ----------
        x : 1-D float array (one sample)
        """
        row = x.reshape(1, -1)
        proba = self.predict_proba(row)[0]
        idx = int(np.argmax(proba))
        label = self.classes_[idx]
        conf = float(proba[idx])
        return label, conf

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def macro_f1(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Compute macro-F1 with bootstrap 95% CI.

        Returns
        -------
        (point_estimate, lower_95, upper_95)
        """
        from sklearn.metrics import f1_score

        preds = self.predict(X)
        point = float(f1_score(y, preds, average="macro", zero_division=0))

        rng = np.random.default_rng(seed)
        n = len(y)
        scores = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            s = float(f1_score(y[idx], preds[idx], average="macro", zero_division=0))
            scores.append(s)

        scores.sort()
        lo = float(np.percentile(scores, 2.5))
        hi = float(np.percentile(scores, 97.5))
        return point, lo, hi

    def feature_importances(self) -> np.ndarray:
        """Return feature importance array (gain), shape (n_features,)."""
        return self._model.feature_importances_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle-serialise to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> LightGBMClassifier:
        """Load a previously saved :class:`LightGBMClassifier`."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LightGBMClassifier, got {type(obj)}")
        return obj
