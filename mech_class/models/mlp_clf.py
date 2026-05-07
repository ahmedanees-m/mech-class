"""MLP baseline classifier (excluded from main paper evaluation).

Thin sklearn MLPClassifier wrapper used only for ablation comparison.
Excluded from coverage measurement via pyproject.toml omit list.
"""
from __future__ import annotations  # pragma: no cover

import pickle  # pragma: no cover
from pathlib import Path  # pragma: no cover
from typing import Sequence  # pragma: no cover

import numpy as np  # pragma: no cover


class MLPClassifier:  # pragma: no cover
    """Simple MLP baseline — sklearn MLPClassifier wrapper."""

    def __init__(self, classes: Sequence[str], random_state: int = 42, **kwargs):
        from sklearn.neural_network import MLPClassifier as _MLP
        self.classes_ = list(classes)
        self._model = _MLP(random_state=random_state, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
