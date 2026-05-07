"""Shared pytest fixtures for mech-class test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_evidence_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "accession": "P001",
                "tier_a": "DSB_NUCLEASE",
                "weight": 1.0,
                "source": "M-CSA",
                "tier_b": "N1_CRISPR_Cas",
                "composite_flag": False,
            },
            {
                "accession": "P002",
                "tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
                "weight": 0.7,
                "source": "TnPedia_ISfinder",
                "tier_b": "B3_Programmable_Recombinase",
                "composite_flag": True,
            },
            {
                "accession": "P003",
                "tier_a": "TRANSPOSASE",
                "weight": 0.9,
                "source": "CRISPRCasdb",
                "tier_b": "T1_DDE_Transposase",
                "composite_flag": False,
            },
        ]
    )


@pytest.fixture
def small_feature_matrix(rng: np.random.Generator):
    n, d = 30, 20
    X = rng.normal(size=(n, d)).astype(np.float32)
    classes = ["DSB_NUCLEASE", "DSB_FREE_TRANSEST_RECOMBINASE", "TRANSPOSASE"]
    y = np.array(classes * (n // 3))
    return X, y
