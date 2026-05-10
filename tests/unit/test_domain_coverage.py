"""Coverage tests for mech_class.features.domain - paths not covered by
test_domain_features.py.  Covers:
  - fetch_pfam_hits_uniprot success path (lines 128-133)
  - fetch_pfam_hits_uniprot retry logic (lines 135-138)
  - fetch_pfam_hits_uniprot all retries exhausted (line 138 return [])
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mech_class.features.domain import fetch_pfam_hits_uniprot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uniprot_response(pfam_ids: list[str]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "uniProtKBCrossReferences": [{"database": "Pfam", "id": pid} for pid in pfam_ids]
        + [{"database": "Gene3D", "id": "1.10.10.10"}]
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# fetch_pfam_hits_uniprot - success path (lines 126-133)
# ---------------------------------------------------------------------------


class TestFetchPfamHitsUniprotSuccess:
    def test_returns_pfam_list_on_success(self):
        """Lines 126-133: successful HTTP response -> returns Pfam IDs."""
        mock_resp = _make_uniprot_response(["PF13395", "PF01548"])
        with patch("mech_class.features.domain.requests.get", return_value=mock_resp):
            result = fetch_pfam_hits_uniprot("Q99ZW2", timeout=10, retries=0)
        assert result == ["PF13395", "PF01548"]

    def test_filters_non_pfam_references(self):
        """Only Pfam entries from uniProtKBCrossReferences are returned."""
        with patch("mech_class.features.domain.requests.get", return_value=_make_uniprot_response(["PF13395"])):
            result = fetch_pfam_hits_uniprot("Q99ZW2", timeout=10, retries=0)
        assert "Gene3D" not in str(result)
        assert result == ["PF13395"]

    def test_empty_cross_references_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"uniProtKBCrossReferences": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("mech_class.features.domain.requests.get", return_value=mock_resp):
            result = fetch_pfam_hits_uniprot("Q99ZW2", retries=0)
        assert result == []

    def test_no_cross_references_key_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        with patch("mech_class.features.domain.requests.get", return_value=mock_resp):
            result = fetch_pfam_hits_uniprot("Q99ZW2", retries=0)
        assert result == []


# ---------------------------------------------------------------------------
# fetch_pfam_hits_uniprot - retry logic (lines 135-138)
# ---------------------------------------------------------------------------


class TestFetchPfamHitsUniprotRetry:
    def test_retries_once_on_failure_then_succeeds(self):
        """Lines 135-138: first attempt fails, second succeeds."""
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network unreachable")
            return _make_uniprot_response(["PF13395"])

        with patch("mech_class.features.domain.requests.get", side_effect=mock_get), patch("time.sleep"):
            result = fetch_pfam_hits_uniprot("Q99ZW2", timeout=5, retries=1)

        assert result == ["PF13395"]
        assert call_count == 2

    def test_all_retries_exhausted_returns_empty(self):
        """Line 138: all attempts fail -> returns []."""
        with (
            patch("mech_class.features.domain.requests.get", side_effect=ConnectionError("always fails")),
            patch("time.sleep"),
        ):
            result = fetch_pfam_hits_uniprot("Q99ZW2", timeout=5, retries=2)
        assert result == []

    def test_sleep_called_between_retries(self):
        """time.sleep(2) is called between retries."""
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with patch("mech_class.features.domain.requests.get", side_effect=mock_get), patch("time.sleep") as mock_sleep:
            fetch_pfam_hits_uniprot("Q99ZW2", timeout=5, retries=2)

        # 3 total attempts (attempt 0, 1, 2) -> sleep called after attempts 0 and 1
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2)

    def test_no_retry_on_zero_retries(self):
        """retries=0 -> only one attempt, no sleep."""
        with (
            patch("mech_class.features.domain.requests.get", side_effect=ConnectionError("fail")),
            patch("time.sleep") as mock_sleep,
        ):
            result = fetch_pfam_hits_uniprot("Q99ZW2", retries=0)
        assert result == []
        mock_sleep.assert_not_called()

    def test_http_error_triggers_retry(self):
        """HTTP 500 triggers retry."""
        import requests as req_lib

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError("500")
                return resp
            return _make_uniprot_response(["PF13395"])

        with patch("mech_class.features.domain.requests.get", side_effect=mock_get), patch("time.sleep"):
            result = fetch_pfam_hits_uniprot("Q99ZW2", retries=1)

        assert result == ["PF13395"]
