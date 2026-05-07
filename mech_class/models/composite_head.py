"""Binary IS110 composite head — LightGBM-based classifier with FP guard.

Wraps ``lgb.LGBMClassifier`` for binary classification (composite=1 vs 0).
Exposes threshold-controllable predict methods and a false-positive rate helper.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGBM_AVAILABLE = False


class CompositeHead:
    """Binary composite-architecture head.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    threshold : float
        Probability threshold for composite=True.
    n_estimators : int
        Number of boosting rounds.
    """

    def __init__(
        self,
        seed: int = 42,
        threshold: float = 0.5,
        n_estimators: int = 200,
    ):
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm is required: pip install mech-class[dev]")
        self._seed = seed
        self._threshold = threshold
        self._model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            random_state=seed,
            verbose=-1,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> CompositeHead:
        """Fit the binary classifier.

        Parameters
        ----------
        X : (n, d) float32 array
        y : (n,) int array of 0/1 labels
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Return P(composite=1) for every row, shape (n_samples,)."""
        proba_matrix = self._model.predict_proba(X)  # (n, 2)
        return proba_matrix[:, 1].astype(np.float32)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0/1) using the current threshold."""
        proba = self.predict_proba_batch(X)
        return (proba >= self._threshold).astype(np.int32)

    # ------------------------------------------------------------------
    # Single-sample prediction
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> tuple[bool, list[str]]:
        """Predict composite flag for a single sample.

        Parameters
        ----------
        x : 1-D float array (one sample)

        Returns
        -------
        composite : bool
        evidence : list[str]
            Reasoning tokens (empty in base implementation; override for richer logic).
        """
        row = x.reshape(1, -1)
        prob = float(self.predict_proba_batch(row)[0])
        composite = prob >= self._threshold
        evidence: list[str] = []
        if composite:
            evidence.append(f"composite_prob={prob:.3f}")
        return bool(composite), evidence

    # ------------------------------------------------------------------
    # Threshold control
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> None:
        """Set the probability threshold for composite=True.

        Parameters
        ----------
        threshold : float in [0, 1]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def fp_rate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return false-positive rate on the negative class.

        Parameters
        ----------
        X : (n, d) float array
        y : (n,) int array of 0/1 true labels

        Returns
        -------
        float — FP / (FP + TN)  (0.0 if no negatives)
        """
        y = np.asarray(y)
        neg_mask = y == 0
        if not neg_mask.any():
            return 0.0
        preds = self.predict_batch(X[neg_mask])
        fp = int((preds == 1).sum())
        tn = int((preds == 0).sum())
        denom = fp + tn
        return fp / denom if denom > 0 else 0.0

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
    def load(cls, path: str | Path) -> CompositeHead:
        """Load a previously saved :class:`CompositeHead`."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CompositeHead, got {type(obj)}")
        return obj
