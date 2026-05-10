"""Unit tests for domain feature encoder and IS110 composite detection.

Uses the actual API: extract_domain_features(pfam_hits) returning a 26-dim vector.
"""

from __future__ import annotations

import numpy as np

from mech_class.features.domain import (
    PFAM_WHITELIST,
    RUVC_DEDD_PF,
    TNP_SERINE_PF,
    extract_domain_features,
)

RUVC_PF = RUVC_DEDD_PF  # "PF01548" IS110 N-terminal
SER_PF = TNP_SERINE_PF  # "PF02371" IS110 C-terminal
CRISPR_PF = "PF13395"  # dom_0 - HNH endonuclease


class TestExtractDomainFeatures:
    """Tests for extract_domain_features(pfam_hits) -> 26-dim float32 vector."""

    def test_returns_correct_dim(self):
        vec = extract_domain_features([])
        assert len(vec) == 26

    def test_returns_float32(self):
        vec = extract_domain_features([])
        assert vec.dtype == np.float32

    def test_empty_gives_zeros(self):
        vec = extract_domain_features([])
        assert np.all(vec == 0.0)

    def test_none_gives_zeros(self):
        vec = extract_domain_features(None)
        assert np.all(vec == 0.0)

    def test_known_pfam_sets_bit(self):
        """Every whitelist entry should set its corresponding dom_i bit."""
        for i, pfam in enumerate(PFAM_WHITELIST):
            vec = extract_domain_features([pfam])
            assert vec[i] == 1.0, f"{pfam} (dom_{i}) should be 1.0"

    def test_unknown_domain_ignored(self):
        """Unknown Pfam accession must not raise and must leave the vector unchanged."""
        vec_known = extract_domain_features([RUVC_PF])
        vec_extra = extract_domain_features([RUVC_PF, "PF99999_FAKE"])
        assert np.array_equal(vec_known, vec_extra), "Unknown Pfam should be silently ignored - vector must not change"

    # IS110 composite flag (dom_23)

    def test_is110_composite_requires_both_domains(self):
        """dom_23 = 1.0 only when both PF01548 AND PF02371 are present."""
        assert extract_domain_features([RUVC_PF])[23] == 0.0, "Only RUVC_PF -> not composite"
        assert extract_domain_features([SER_PF])[23] == 0.0, "Only SER_PF  -> not composite"
        assert extract_domain_features([])[23] == 0.0, "Empty       -> not composite"
        assert extract_domain_features([RUVC_PF, SER_PF])[23] == 1.0, "Both        -> composite"

    def test_is110_composite_differs_from_single(self):
        """The IS110 composite vector must differ from the single-domain vector."""
        vec_both = extract_domain_features([RUVC_PF, SER_PF])
        vec_one = extract_domain_features([RUVC_PF])
        assert not np.array_equal(vec_both, vec_one)

    # Single-domain flag (dom_25)

    def test_single_domain_flag_true(self):
        """dom_25 = 1.0 when exactly one whitelist Pfam is present."""
        vec = extract_domain_features([CRISPR_PF])
        assert vec[25] == 1.0, "Single whitelist hit -> dom_25 should be 1.0"

    def test_single_domain_flag_false_for_two(self):
        """dom_25 = 0.0 when two whitelist Pfams are present."""
        vec = extract_domain_features([RUVC_PF, SER_PF])
        assert vec[25] == 0.0, "Two whitelist hits -> dom_25 should be 0.0"

    def test_single_domain_flag_false_for_zero(self):
        """dom_25 = 0.0 when no whitelist Pfam is present."""
        vec = extract_domain_features(["PF99999_FAKE"])
        assert vec[25] == 0.0, "No whitelist hits -> dom_25 should be 0.0"

    # dom_24 (editor fusion - reserved)

    def test_dom24_always_zero(self):
        """dom_24 (editor fusion) is reserved; always 0 in v1.0."""
        vec = extract_domain_features([RUVC_PF, SER_PF, CRISPR_PF])
        assert vec[24] == 0.0, "dom_24 should always be 0 (reserved for v1.1)"

    # Probe vectors (golden reference)

    def test_cas9_probe_domain_vector(self):
        """SpCas9 canonical Pfam set: 6 domains -> dom_0,1,2,3,19,20 set; dom_25=0."""
        cas9_pfam = ["PF13395", "PF18541", "PF16595", "PF18516", "PF16592", "PF16593"]
        vec = extract_domain_features(cas9_pfam)
        # dom_0 (PF13395), dom_1 (PF18541), dom_2 (PF16595), dom_3 (PF18516)
        for dom_idx in [0, 1, 2, 3]:
            assert vec[dom_idx] == 1.0, f"dom_{dom_idx} should be 1.0 for SpCas9"
        # Not single-domain
        assert vec[25] == 0.0

    def test_tn5_probe_domain_vector(self):
        """Tn5 (PF01609 = dom_8): single domain -> dom_25 = 1.0."""
        vec = extract_domain_features(["PF01609"])
        assert vec[8] == 1.0, "PF01609 -> dom_8 should be 1.0"
        assert vec[25] == 1.0, "Single whitelist hit -> dom_25 should be 1.0"


# get_pfam_accessions


class TestGetPfamAccessions:
    def test_returns_list_of_23(self):
        from mech_class.features.domain import get_pfam_accessions

        pf = get_pfam_accessions()
        assert len(pf) == 23

    def test_returns_copy_not_singleton(self):
        """Mutating the returned list must not corrupt the module constant."""
        from mech_class.features.domain import get_pfam_accessions

        pf1 = get_pfam_accessions()
        pf2 = get_pfam_accessions()
        pf1.clear()  # mutate the copy
        assert len(pf2) == 23, "get_pfam_accessions() should return an independent copy"


# fetch_pfam_hits_uniprot (error fallback only)


class TestFetchPfamHitsUniprot:
    def test_invalid_accession_returns_empty_list(self):
        """Invalid accession -> 404/error -> empty list (no exception raised)."""
        from mech_class.features.domain import fetch_pfam_hits_uniprot

        result = fetch_pfam_hits_uniprot("INVALID_ACCESSION_XYZ_99999", timeout=5, retries=0)
        assert isinstance(result, list)
        assert result == []

    def test_returns_list_type_always(self):
        """Function must always return a list, never raise."""
        from mech_class.features.domain import fetch_pfam_hits_uniprot

        result = fetch_pfam_hits_uniprot("UNREACHABLE_9999", timeout=1, retries=0)
        assert isinstance(result, list)
