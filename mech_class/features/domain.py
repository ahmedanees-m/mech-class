"""F_domain channel: Pfam binary features + composite-architecture flags.

Binary presence vector over the 23 primary Pfam families from training
(dom_0..dom_22), plus three derived flags (dom_23..dom_25):

  dom_23 — IS110 composite: PF01548 AND PF02371 present in same protein
  dom_24 — Editor fusion:   reserved (Cas9-nickase + RT co-occurrence)
  dom_25 — Single-domain:   exactly one whitelist hit present

Total: 26 binary features (dom_0..dom_25).

The PFAM_WHITELIST order is FIXED to match the training feature_matrix.parquet
columns (dom_0..dom_22).  Do not reorder.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import requests

# ── Pfam whitelist ────────────────────────────────────────────────────────────
# Hardcoded to match training (scripts/13_compute_domain_features.py row).
# DO NOT reorder — dom_i indices are burned into all model.pkl files.
PFAM_WHITELIST: list[str] = [
    "PF13395",  # dom_0  HNH_4 (HNH endonuclease)
    "PF18541",  # dom_1  RuvC_III (RuvC-like, Cas12/Cpf1 family)
    "PF16595",  # dom_2  Cas9_RuvC-II
    "PF18516",  # dom_3  Cas9_HNH_2
    "PF01548",  # dom_4  DEDD_Tnp_IS110 (IS110 N-terminal RuvC-fold)
    "PF02371",  # dom_5  Transposase_20 (IS110 C-terminal serine domain)
    "PF07282",  # dom_6  Cas12f1-like_TNB (TnpB/Fanzor OMEGA-nuclease)
    "PF00665",  # dom_7  rve (integrase core, DDE transposase)
    "PF01609",  # dom_8  DDE_Tnp_1_7 (Tn5-like DDE transposase)
    "PF13586",  # dom_9  DDE_Tnp_1_4
    "PF08721",  # dom_10 DDE_3 (Tc1/mariner transposase)
    "PF11426",  # dom_11 DDE_Tnp_IS1595
    "PF05621",  # dom_12 Transposase_21 (IS3/IS150/IS904 family)
    "PF00589",  # dom_13 Phage_integrase (tyrosine recombinase / Cre)
    "PF00239",  # dom_14 Resolvase (serine recombinase)
    "PF07508",  # dom_15 Recombinase (large serine integrase, Bxb1)
    "PF01844",  # dom_16 HHH (helix-hairpin-helix, accessory)
    "PF02486",  # dom_17 CP_lyase (CRISPR-associated)
    "PF18061",  # dom_18 Cas9_NTD
    "PF16592",  # dom_19 Cas9_bridge_helix
    "PF16593",  # dom_20 Cas9_PAM_int
    "PF13639",  # dom_21 RuvX (MutH-like)
    "PF03377",  # dom_22 IS200_IS605 (IS200/IS605 TnpB)
]

# Key domain constants
RUVC_DEDD_PF = "PF01548"   # IS110 N-terminal (dom_4)
TNP_SERINE_PF = "PF02371"  # IS110 C-terminal (dom_5)


def get_pfam_accessions() -> list[str]:
    """Return ordered whitelist (23 entries, dom_0..dom_22)."""
    return list(PFAM_WHITELIST)


def extract_domain_features(
    pfam_hits: Optional[list[str]] = None,
    *,
    sequence: str = "",   # kept for API compatibility; not used
) -> np.ndarray:
    """Build a 26-dim domain feature vector (dom_0..dom_25).

    Parameters
    ----------
    pfam_hits : list[str], optional
        Pfam accession IDs present in the protein.
        Returns a zero vector if None or empty.
    sequence : str
        Not used; kept for backward-compatibility.

    Returns
    -------
    np.ndarray, shape (26,), dtype float32
    """
    vec = np.zeros(26, dtype=np.float32)
    if not pfam_hits:
        return vec

    pfam_set = set(pfam_hits)
    wl_hits: list[str] = []

    for i, acc in enumerate(PFAM_WHITELIST):
        if acc in pfam_set:
            vec[i] = 1.0
            wl_hits.append(acc)

    # dom_23: IS110 composite
    vec[23] = 1.0 if (RUVC_DEDD_PF in pfam_set and TNP_SERINE_PF in pfam_set) else 0.0
    # dom_24: editor fusion (reserved; always 0 in v1.0)
    # dom_25: single-domain flag
    vec[25] = 1.0 if len(wl_hits) == 1 else 0.0

    return vec


def fetch_pfam_hits_uniprot(
    accession: str,
    *,
    timeout: int = 15,
    retries: int = 2,
) -> list[str]:
    """Fetch Pfam cross-references for a UniProt accession via REST API.

    Parameters
    ----------
    accession : str
        UniProt accession (e.g. 'Q99ZW2').
    timeout : int
        Request timeout in seconds.
    retries : int
        Number of retry attempts on failure.

    Returns
    -------
    list[str]
        Pfam accession IDs (e.g. ['PF13395', 'PF18541']).
        Returns empty list on any error.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            pfam: list[str] = []
            for ref in data.get("uniProtKBCrossReferences", []):
                if ref.get("database") == "Pfam":
                    pfam.append(ref["id"])
            return pfam
        except Exception:
            if attempt < retries:
                import time; time.sleep(2)
    return []


def build_domain_feature_matrix(
    accessions: list[str],
    duckdb_path: str = "/data/graphs/atlas.duckdb",
) -> np.ndarray:
    """Build N × 26 domain feature matrix using ATLAS DuckDB.

    Parameters
    ----------
    accessions : list[str]
        UniProt accessions (must be in ATLAS).
    duckdb_path : str
        Path to ATLAS DuckDB (from Paper 1).

    Returns
    -------
    np.ndarray, shape (N, 26), dtype float32
    """
    import duckdb
    from collections import defaultdict

    matrix    = np.zeros((len(accessions), 26), dtype=np.float32)
    acc_to_idx = {acc: i for i, acc in enumerate(accessions)}

    con = duckdb.connect(duckdb_path, read_only=True)
    acc_list_sql = ", ".join(f"'{a}'" for a in accessions)
    rows = con.execute(f"""
        SELECT p.accession, d.accession AS pfam_acc
        FROM edges e
        JOIN nodes_protein p ON p.id = e.source_id AND e.source_type = 'Protein'
        JOIN nodes_domain  d ON d.id = e.target_id AND e.target_type = 'Domain'
        WHERE p.accession IN ({acc_list_sql})
    """).fetchall()
    con.close()

    protein_domains: dict[str, list[str]] = defaultdict(list)
    for prot_acc, pfam_acc in rows:
        protein_domains[prot_acc].append(pfam_acc)

    for acc, pfam_hits in protein_domains.items():
        idx = acc_to_idx.get(acc)
        if idx is not None:
            matrix[idx] = extract_domain_features(pfam_hits)

    return matrix
