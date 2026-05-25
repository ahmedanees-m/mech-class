"""Integration tests for Predictor.load() → predict_from_sequence() pipeline.

Covers mech_class/api.py end-to-end using the 10-protein smoke test probes
with canonical_pfam supplied to bypass UniProt REST lookup (avoids annotation
drift between training-time Atlas and current UniProt).

Skip condition: trained model files not found at /data/models/tier_a/model.pkl.
  → All tests skip gracefully in CI environments without model artifacts.
  → On VM: pytest runs fully; all 10 probes expected to PASS.

Canonical Pfam rationale:
  Training used GENOME-ATLAS Pfam annotations. Current UniProt may differ:
  e.g. Fanzor Q8I6T1 has PF07282 in Atlas vs PF18297 in UniProt. canonical_pfam
  ensures domain features match training distribution exactly.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

MODEL_DIR = Path("/data/models")

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tier_a" / "model.pkl").exists(),
    reason="Trained models not found at /data/models — run on VM after training",
)

# Detect whether ESM-2 is installed (fair-esm package)
_ESM2_AVAILABLE = importlib.util.find_spec("esm") is not None

# ── Probe definitions ─────────────────────────────────────────────────────────
# Same canonical_pfam lists as scripts/50_predictor_smoke_test.py.
# Sequences are fetched from UniProt REST inside each test (one request per probe).

_PROBES = [
    {
        "label": "IS110 (holdout)",
        "accession": "A0A7C9VKZ0",
        "expected_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_conf": 0.60,
        "composite": True,
        "canonical_pfam": ["PF01548", "PF02371"],
        "requires_esm2": True,  # domain features alone insufficient; ESM-2 required
    },
    {
        "label": "Fanzor SpFanzor1 (holdout)",
        "accession": "Q8I6T1",
        "expected_tier_a": "DSB_NUCLEASE",
        "min_conf": 0.70,
        "composite": False,
        "canonical_pfam": ["PF07282"],
    },
    {
        "label": "SpCas9 (holdout; composite FP documented)",
        "accession": "Q99ZW2",
        "expected_tier_a": "DSB_NUCLEASE",
        "min_conf": 0.60,
        "composite": None,  # xfail; composite FP documented in MODEL_CARD.md
        "canonical_pfam": ["PF13395", "PF18541", "PF16595", "PF18516", "PF16592", "PF16593"],
    },
    {
        "label": "Bxb1 integrase (holdout)",
        "accession": "Q9B086",
        "expected_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_conf": 0.60,
        "composite": False,
        "canonical_pfam": ["PF07508", "PF00239"],
        "requires_esm2": True,  # Bxb1 integrase needs seq embedding for correct Tier-A
    },
    {
        "label": "Tn5 transposase (holdout)",
        "accession": "Q46731",
        "expected_tier_a": "TRANSPOSASE",
        "min_conf": 0.60,
        "composite": False,
        "canonical_pfam": ["PF01609"],
        "requires_esm2": True,  # DDE transposase domain alone insufficient without ESM-2
    },
    {
        "label": "Cre recombinase (in-distribution)",
        "accession": "P06956",
        "expected_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_conf": 0.60,
        "composite": False,
        "canonical_pfam": ["PF00589"],
    },
    {
        "label": "AsCpf1 / Cas12a",
        "accession": "Q0P897",
        "expected_tier_a": "DSB_NUCLEASE",
        "min_conf": 0.50,
        "composite": None,
        "canonical_pfam": ["PF13395", "PF18541"],
    },
    {
        "label": "Lambda integrase (phage Int)",
        "accession": "P03700",
        "expected_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_conf": 0.50,
        "composite": False,
        "canonical_pfam": ["PF00589"],
    },
    {
        "label": "IS10 transposase (Tn10, E. coli)",
        "accession": "P0CF64",
        "expected_tier_a": "TRANSPOSASE",
        "min_conf": 0.50,
        "composite": None,
        "canonical_pfam": ["PF01609"],
        "requires_esm2": True,  # DDE transposase; ESM-2 needed to separate from nuclease
    },
    {
        "label": "IscB-like TnpB (Cas12f-like, H. pylori)",
        "accession": "P75538",
        "expected_tier_a": "DSB_NUCLEASE",
        "min_conf": 0.50,
        "composite": None,
        "canonical_pfam": ["PF07282"],
    },
    {
        "label": "ISCro4 OOD gate probe (D2TGM5, Citrobacter rodentium)",
        "accession": "D2TGM5",
        "expected_tier_a": "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_conf": 0.90,  # gate floor; conf = max(ML_DSB_FREE, 0.90)
        "composite": True,
        "canonical_pfam": ["PF01548", "PF02371"],
        # requires_esm2=False: Tier-A IS110 hard gate fires on PF01548∧PF02371 alone.
        # D2TGM5 is OOD (not in training set → zero ESM-2 embedding at inference).
        # Without gate, ML predicts DSB_NUCLEASE P≈0.57 (IS110 OOD failure, v0.5.2 bug).
        # Gate overrides ML output → DSB_FREE_TRANSEST_RECOMBINASE, tier_a_gate_override=True.
        # Canonical name: ISCro4 (UniProt D2TGM5 + Pelea 2026 Science adz1884).
        # "IS622" is the deprecated preprint label (Perry 2025 bioRxiv 2025.05.14.653916).
    },
]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def predictor():
    """Load the trained Predictor once per module (ESM-2 loaded lazily on first predict)."""
    from mech_class.api import Predictor

    return Predictor.load(MODEL_DIR)


def _fetch_sequence(accession: str) -> str:
    """Fetch UniProt FASTA sequence. Skips test on network failure."""
    import urllib.request

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            fasta = resp.read().decode("utf-8")
        lines = [ln for ln in fasta.strip().split("\n") if not ln.startswith(">")]
        seq = "".join(lines).strip()
        if not seq:
            pytest.skip(f"Empty sequence for {accession}")
        return seq
    except Exception as e:
        pytest.skip(f"Network unavailable for {accession}: {e}")


# ── Parameterised Tier-A tests ────────────────────────────────────────────────


@pytest.mark.parametrize("probe", _PROBES, ids=[p["label"] for p in _PROBES])
def test_tier_a(predictor, probe):
    """Tier-A must match expected class with minimum confidence.

    Probes marked ``requires_esm2=True`` are skipped when fair-esm is not
    installed — the Tier-A model relies heavily on ESM-2 sequence embeddings
    (640 of 1953 features) and domain features alone are insufficient for these
    ambiguous classes without ESM-2 context.
    """
    if probe.get("requires_esm2") and not _ESM2_AVAILABLE:
        pytest.skip(
            f"{probe['label']}: requires fair-esm for correct Tier-A prediction (install with: pip install fair-esm)"
        )
    seq = _fetch_sequence(probe["accession"])
    pred = predictor.predict_from_sequence(probe["accession"], seq, pfam_hits=probe["canonical_pfam"])
    assert pred.tier_a == probe["expected_tier_a"], (
        f"{probe['label']}: expected tier_a={probe['expected_tier_a']!r}, got {pred.tier_a!r}"
    )
    assert pred.tier_a_confidence >= probe["min_conf"], (
        f"{probe['label']}: confidence {pred.tier_a_confidence:.3f} < {probe['min_conf']}"
    )


@pytest.mark.parametrize(
    "probe",
    [p for p in _PROBES if p["composite"] is not None],
    ids=[p["label"] for p in _PROBES if p["composite"] is not None],
)
def test_composite(predictor, probe):
    """Composite flag must match expected value (True/False) for non-None probes."""
    seq = _fetch_sequence(probe["accession"])
    pred = predictor.predict_from_sequence(probe["accession"], seq, pfam_hits=probe["canonical_pfam"])
    assert pred.composite == probe["composite"], (
        f"{probe['label']}: expected composite={probe['composite']}, got {pred.composite} (P={pred.composite_prob:.3f})"
    )


@pytest.mark.xfail(
    reason="SpCas9 composite=True (FP, P≈0.753). Documented in MODEL_CARD.md Limitation 3. "
    "Composite head over-fires for proteins with ≥4 whitelist Pfam domains."
)
def test_cas9_composite_false(predictor):
    """SpCas9 should ideally be composite=False (FP documented, xfail accepted)."""
    seq = _fetch_sequence("Q99ZW2")
    pred = predictor.predict_from_sequence(
        "Q99ZW2",
        seq,
        pfam_hits=["PF13395", "PF18541", "PF16595", "PF18516", "PF16592", "PF16593"],
    )
    assert not pred.composite, f"SpCas9 composite FP: P={pred.composite_prob:.3f}"


# ── ISCro4/D2TGM5 Tier-A gate override test ───────────────────────────────────


def test_iscro4_tier_a_gate_override(predictor):
    """D2TGM5 (ISCro4): Tier-A IS110 hard gate must fire → tier_a_gate_override=True.

    This is the canonical OOD probe for the v0.5.2 IS110 gate fix. D2TGM5 has no
    pre-computed ESM-2 embedding (not in training set) → ML alone predicts
    DSB_NUCLEASE P≈0.57 (OOD failure). The gate overrides this to DSB_FREE with
    conf ≥ 0.90 whenever PF01548 ∧ PF02371 are both present.
    """
    seq = _fetch_sequence("D2TGM5")
    pred = predictor.predict_from_sequence("D2TGM5", seq, pfam_hits=["PF01548", "PF02371"])
    assert pred.tier_a == "DSB_FREE_TRANSEST_RECOMBINASE", (
        f"ISCro4 gate must force DSB_FREE_TRANSEST_RECOMBINASE; got {pred.tier_a!r}"
    )
    assert pred.tier_a_gate_override is True, (
        "ISCro4 gate must set tier_a_gate_override=True (OOD ML output overridden)"
    )
    assert pred.tier_a_confidence >= 0.90, f"ISCro4 gate floor is 0.90; got {pred.tier_a_confidence:.3f}"
    assert pred.composite is True, "ISCro4 PF01548+PF02371 → composite=True"


# ── Prediction object structural tests ───────────────────────────────────────


def test_prediction_has_required_fields(predictor):
    """Prediction Pydantic model must expose all required fields."""
    seq = _fetch_sequence("P06956")  # Cre — reliable in-distribution probe
    pred = predictor.predict_from_sequence("P06956", seq, pfam_hits=["PF00589"])

    assert isinstance(pred.accession, str)
    assert isinstance(pred.sequence_length, int)
    assert pred.sequence_length > 0
    assert isinstance(pred.tier_a, str)
    assert isinstance(pred.tier_a_confidence, float)
    assert isinstance(pred.composite, bool)
    assert isinstance(pred.composite_prob, float)
    assert isinstance(pred.pfam_hits, list)
    assert isinstance(pred.channels_used, list)


def test_prediction_confidence_in_range(predictor):
    """tier_a_confidence must be in [0, 1]."""
    seq = _fetch_sequence("P06956")
    pred = predictor.predict_from_sequence("P06956", seq, pfam_hits=["PF00589"])
    assert 0.0 <= pred.tier_a_confidence <= 1.0


def test_prediction_channels_used_nonempty(predictor):
    """At minimum F_domain should be listed when pfam_hits is supplied."""
    seq = _fetch_sequence("P06956")
    pred = predictor.predict_from_sequence("P06956", seq, pfam_hits=["PF00589"])
    assert len(pred.channels_used) >= 1, "channels_used should not be empty"
    assert "F_domain" in pred.channels_used


def test_prediction_without_pfam_uses_zero_domain(predictor):
    """Calling without pfam_hits should not crash (F_domain zero-filled)."""
    seq = _fetch_sequence("P06956")
    pred = predictor.predict_from_sequence("P06956", seq, pfam_hits=[])
    assert isinstance(pred.tier_a, str)
    assert "F_domain" not in pred.channels_used


def test_summary_method_returns_string(predictor):
    """Prediction.summary() must return a non-empty string."""
    seq = _fetch_sequence("P06956")
    pred = predictor.predict_from_sequence("P06956", seq, pfam_hits=["PF00589"])
    summary = pred.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "P06956" in summary


# ── Predictor.load() error handling ──────────────────────────────────────────


def test_load_raises_on_missing_dir(tmp_path):
    """Predictor.load() with non-existent directory must raise FileNotFoundError."""
    from mech_class.api import Predictor

    with pytest.raises(FileNotFoundError, match="Tier-A model not found"):
        Predictor.load(tmp_path / "does_not_exist")


def test_download_stub_raises_runtime_error(tmp_path):
    """_download_from_zenodo must raise RuntimeError (stub; Zenodo not yet live)."""
    from mech_class.api import _download_from_zenodo

    with pytest.raises(RuntimeError, match="Zenodo"):
        _download_from_zenodo(tmp_path)
