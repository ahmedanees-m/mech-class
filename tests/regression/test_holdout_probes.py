"""Regression tests -- pre-registered holdout probe criteria (corrected accessions).

These tests run against a trained model. They are skipped
unless /data/models/tier_a/model.pkl exists.

Accession corrections (2026-05-06):
  Bxb1: Q8VVR2 (wrong) -> Q9B086 (Mycobacteriophage Bxb1 integrase, 500 AA)
  Tn5:  P00509 (wrong)  -> Q46731 (E. coli Tn5 transposase)
  See LABEL_PROVENANCE.md Data Pipeline Corrections.

Pre-registered success criteria (Tier-A only):
  IS110   : tier_a == DSB_FREE_TRANSEST_RECOMBINASE, confidence >= 0.60, composite=True
  Fanzor  : tier_a == DSB_NUCLEASE, confidence >= 0.70
  Cas9    : tier_a == DSB_NUCLEASE, confidence >= 0.60
  Bxb1    : tier_a == DSB_FREE_TRANSEST_RECOMBINASE, confidence >= 0.60
  Tn5     : tier_a == TRANSPOSASE, confidence >= 0.60

Tier-B NOT tested: Tier-B sub-classifiers return UNKNOWN for all probes due to
small per-class training N (< 3 for TRANSPOSASE; 39 for DSB_NUCLEASE). This is
explicitly acceptable per label_taxonomy.yaml (Tier-B is supplementary / ungated).
Tier-B tests are kept as xfail to document the known limitation.

Note on composite head: SpCas9 (Q99ZW2) lacks PF01548/PF02371, so the domain
gate forces composite=False even though the raw ML score is 0.753. See MODEL_CARD.md.

Note on Cre (P06956): found in training set; cannot be used as OOD holdout.
No test for Cre.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_DIR = Path("/data/models")
HOLDOUT_FEAT = Path("/data/validation/holdout_features.parquet")

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tier_a" / "model.pkl").exists(),
    reason="Trained model not found at /data/models/tier_a/model.pkl -- run after training",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    with open(MODEL_DIR / "tier_a" / "model.pkl", "rb") as f:
        ta = pickle.load(f)
    with open(MODEL_DIR / "composite_head" / "model.pkl", "rb") as f:
        comp = pickle.load(f)
    return ta["model"], ta["feature_cols"], ta["label_encoder"], comp["model"], comp["feature_cols"]


def _predict(accession: str, models):
    lgbm_a, feat_cols, le_a, lgbm_comp, comp_feat_cols = models
    df = pd.read_parquet(HOLDOUT_FEAT)
    row = df[df["uniprot_acc"] == accession]
    assert not row.empty, f"{accession} not in holdout_features.parquet"
    x = np.zeros(len(feat_cols), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        if c in row.columns:
            x[i] = float(row.iloc[0][c])
    x_df = pd.DataFrame([x], columns=feat_cols)
    proba_a = lgbm_a.predict_proba(x_df)[0]
    pred_idx = int(np.argmax(proba_a))
    tier_a = le_a.inverse_transform([pred_idx])[0]
    conf = float(proba_a[pred_idx])
    x_comp = x_df[comp_feat_cols] if comp_feat_cols else x_df
    comp_proba = lgbm_comp.predict_proba(x_comp)[0]
    gate_pass = bool(x_df["dom_23"].iloc[0] >= 0.5)  # dom_23 = PF01548 and PF02371
    composite = gate_pass and bool(comp_proba[1] >= 0.5)
    return tier_a, conf, composite


# ---------------------------------------------------------------------------
# Tier-A tests (pre-registered, must pass)
# ---------------------------------------------------------------------------


def test_is110_tier_a(models):
    tier_a, conf, composite = _predict("A0A7C9VKZ0", models)
    assert tier_a == "DSB_FREE_TRANSEST_RECOMBINASE", f"IS110 tier_a: {tier_a}"
    assert conf >= 0.60, f"IS110 confidence {conf:.3f} < 0.60"
    assert composite, "IS110 should have composite flag"


def test_fanzor_tier_a(models):
    tier_a, conf, _ = _predict("Q8I6T1", models)
    assert tier_a == "DSB_NUCLEASE", f"Fanzor tier_a: {tier_a}"
    assert conf >= 0.70, f"Fanzor confidence {conf:.3f} < 0.70"


def test_cas9_tier_a(models):
    tier_a, conf, _ = _predict("Q99ZW2", models)
    assert tier_a == "DSB_NUCLEASE", f"SpCas9 tier_a: {tier_a}"
    assert conf >= 0.60, f"SpCas9 confidence {conf:.3f} < 0.60"


def test_bxb1_tier_a(models):
    """Bxb1 integrase -- accession corrected Q8VVR2 -> Q9B086."""
    tier_a, conf, _ = _predict("Q9B086", models)
    assert tier_a == "DSB_FREE_TRANSEST_RECOMBINASE", f"Bxb1 tier_a: {tier_a}"
    assert conf >= 0.60, f"Bxb1 confidence {conf:.3f} < 0.60"


def test_tn5_tier_a(models):
    """Tn5 transposase -- accession corrected P00509 -> Q46731."""
    tier_a, conf, _ = _predict("Q46731", models)
    assert tier_a == "TRANSPOSASE", f"Tn5 tier_a: {tier_a}"
    assert conf >= 0.60, f"Tn5 confidence {conf:.3f} < 0.60"


# ---------------------------------------------------------------------------
# Tier-B tests (xfail -- UNKNOWN for all probes, small training N)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(  # noqa: E501
    reason="Tier-B UNKNOWN for all probes; sub-class training N too small (acceptable per label_taxonomy.yaml)"
)
def test_fanzor_tier_b(models):
    from mech_class.api import Predictor

    pred = Predictor.load(model_dir=MODEL_DIR).predict_from_accession("Q8I6T1")
    assert pred.tier_b == "N2_Fanzor_OMEGA"


@pytest.mark.xfail(reason="Tier-B UNKNOWN for all probes; sub-class training N too small")
def test_cas9_tier_b(models):
    from mech_class.api import Predictor

    pred = Predictor.load(model_dir=MODEL_DIR).predict_from_accession("Q99ZW2")
    assert pred.tier_b == "N1_CRISPR_Cas"


@pytest.mark.xfail(reason="Tier-B UNKNOWN for all probes; sub-class training N too small")
def test_bxb1_tier_b(models):
    from mech_class.api import Predictor

    pred = Predictor.load(model_dir=MODEL_DIR).predict_from_accession("Q9B086")
    assert pred.tier_b == "B3_Programmable_Recombinase"


@pytest.mark.xfail(reason="Tier-B UNKNOWN for all probes; sub-class training N too small")
def test_tn5_tier_b(models):
    from mech_class.api import Predictor

    pred = Predictor.load(model_dir=MODEL_DIR).predict_from_accession("Q46731")
    assert pred.tier_b == "T1_DDE_Transposase"


# ---------------------------------------------------------------------------
# Composite head tests (domain gate blocks non-IS110 composite calls)
# ---------------------------------------------------------------------------


def test_cas9_composite_false(models):
    _, _, composite = _predict("Q99ZW2", models)
    assert not composite, "SpCas9 lacks PF01548/PF02371; domain gate forces composite=False"
