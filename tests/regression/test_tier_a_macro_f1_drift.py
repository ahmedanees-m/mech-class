"""Regression test — Tier-A macro-F1 ≥ 0.9862 baseline.

Pre-registered success criterion (from holdout_set.yaml §0.5 pre-registration):
  - 5-fold stratified CV mean macro-F1 ≥ 0.9862 on the training feature matrix.

Skip condition: feature_matrix.parquet or trained model not found at expected paths.

This test guards against future code changes that regress the classifier's
performance below the published baseline. It does NOT guard against:
  - Randomness in a single run (CV with seed=42 is deterministic).
  - Overfitting (in-sample metric; CV mitigates but does not eliminate).

Paths (Docker volume layout):
  /data/features/feature_matrix.parquet  — training features (N × 1953 float32)
  /data/models/tier_a/model.pkl          — trained Tier-A LightGBM dict
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

FEATURE_MATRIX = Path("/data/features/feature_matrix.parquet")
MODEL_PATH = Path("/data/models/tier_a/model.pkl")
BASELINE_F1 = 0.9862
N_FOLDS = 5
RANDOM_STATE = 42

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not FEATURE_MATRIX.exists(),
    reason=("Tier-A model or feature matrix not found at /data/models and /data/features — run on VM after training"),
)


@pytest.fixture(scope="module")
def train_data():
    """Load feature matrix and labels from parquet."""
    df = pd.read_parquet(FEATURE_MATRIX)
    # Label column is "tier_a" in the feature matrix (verified from training scripts)
    label_col = "tier_a"
    assert label_col in df.columns, (
        f"Expected '{label_col}' column in feature_matrix.parquet; "
        f"found: {[c for c in df.columns if 'tier' in c.lower()]}"
    )
    with open(MODEL_PATH, "rb") as f:
        ta = pickle.load(f)
    feat_cols = ta["feature_cols"]
    X = df[feat_cols].values.astype(np.float32)
    y = df[label_col].values
    return X, y, ta


def test_tier_a_macro_f1_cv_baseline(train_data):
    """5-fold CV macro-F1 must meet pre-registered baseline of ≥ 0.9862."""
    X, y, ta = train_data
    lgbm = ta["model"]
    le = ta["label_encoder"]

    y_enc = le.transform(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for _fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        import lightgbm as lgb

        fold_model = lgb.LGBMClassifier(**lgbm.get_params())
        fold_model.fit(X_tr, y_tr)
        y_pred = fold_model.predict(X_val)
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        scores.append(fold_f1)

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))
    print(f"\nTier-A CV macro-F1: {mean_f1:.4f} ± {std_f1:.4f}  (folds: {scores})")

    assert mean_f1 >= BASELINE_F1, (
        f"Tier-A CV macro-F1 {mean_f1:.4f} fell below pre-registered baseline {BASELINE_F1}. "
        f"Folds: {[round(s, 4) for s in scores]}"
    )


def test_tier_a_all_three_classes_trained(train_data):
    """Tier-A LabelEncoder must have exactly 3 classes."""
    _, _, ta = train_data
    le = ta["label_encoder"]
    classes = list(le.classes_)
    expected = {
        "DSB_NUCLEASE",
        "DSB_FREE_TRANSEST_RECOMBINASE",
        "TRANSPOSASE",
    }
    assert set(classes) == expected, f"Expected Tier-A classes {expected}, got {set(classes)}"


def test_feature_matrix_column_count(train_data):
    """Feature matrix must have 1953 features (640 seq + 1280 struct + 26 dom + 7 as)."""
    _, _, ta = train_data
    feat_cols = ta["feature_cols"]
    assert len(feat_cols) == 1953, f"Expected 1953 feature columns, got {len(feat_cols)}"


def test_is110_holdout_regression(train_data):
    """IS110 (A0A7C9VKZ0): Tier-A=DSB_FREE, composite=True.

    Uses the trained model directly on IS110's feature vector from the matrix.
    This guards against future model changes that break IS110 reclassification.
    """
    X, _, ta = train_data
    df = pd.read_parquet(FEATURE_MATRIX)
    acc_col = "uniprot_acc" if "uniprot_acc" in df.columns else "accession"
    IS110_ACC = "A0A7C9VKZ0"
    row = df[df[acc_col] == IS110_ACC]
    if row.empty:
        pytest.skip(f"IS110 ({IS110_ACC}) not in feature_matrix.parquet")

    feat_cols = ta["feature_cols"]
    x_row = pd.DataFrame(
        row[feat_cols].values.astype(np.float32),
        columns=feat_cols,
    )
    proba_a = ta["model"].predict_proba(x_row)[0]
    pred_idx = int(np.argmax(proba_a))
    tier_a = ta["label_encoder"].inverse_transform([pred_idx])[0]
    conf = float(proba_a[pred_idx])

    assert tier_a == "DSB_FREE_TRANSEST_RECOMBINASE", (
        f"IS110 holdout regression failed: tier_a={tier_a!r} (expected DSB_FREE_TRANSEST_RECOMBINASE)"
    )
    assert conf >= 0.90, f"IS110 holdout confidence {conf:.3f} < 0.90"
