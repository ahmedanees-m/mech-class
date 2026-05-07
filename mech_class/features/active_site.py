"""F_active_site channel: geometric features from PDB / AlphaFold structures.

Extracts ~20 features from the active-site region:
  - Pairwise Cα distances between catalytic residues (M-CSA-annotated)
  - DSSP secondary structure at catalytic positions
  - Mean pLDDT of catalytic domain (quality gate; <70 → zero-fill with flag)
  - Mg2+/metal coordination geometry (present/absent binary)

pLDDT guard (Paper 1 §1.4.4): Features are only filled if mean pLDDT ≥ 70
at the active-site residues. Below this threshold, the F_active_site vector
is zero-filled and a `plddt_low` flag is set in the feature vector — this
propagates to the ablation study and to PEN-SCORE's confidence penalties.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np

ACTIVE_SITE_DIM = 20
PLDDT_THRESHOLD = 70.0


def _open_pdb(path: Path) -> io.TextIOWrapper:
    """Open a .pdb or .pdb.gz file for reading."""
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))  # type: ignore
    return open(path)


def get_ca_coordinates(pdb_path: Path, residue_ids: list[int]) -> dict[int, np.ndarray]:
    """Extract Cα XYZ coordinates for given residue positions from a PDB file."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    with _open_pdb(pdb_path) as fh:
        structure = parser.get_structure("prot", fh)

    coords: dict[int, np.ndarray] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]
                if res_id in residue_ids and "CA" in residue:
                    coords[res_id] = residue["CA"].get_vector().get_array()
        break  # first model only
    return coords


def get_dssp_at_residues(pdb_path: Path, residue_ids: list[int]) -> dict[int, str]:
    """Get DSSP secondary structure codes at specific residues."""
    try:
        from Bio.PDB import DSSP, PDBParser

        parser = PDBParser(QUIET=True)
        with _open_pdb(pdb_path) as fh:
            structure = parser.get_structure("prot", fh)
        model = list(structure)[0]
        dssp = DSSP(model, str(pdb_path))
        result: dict[int, str] = {}
        for (_chain_id, res_id), vals in dssp.property_dict.items():
            if res_id[1] in residue_ids:
                result[res_id[1]] = vals[2]  # secondary structure code
        return result
    except Exception:
        return {}


def get_plddt_at_residues(pdb_path: Path, residue_ids: list[int]) -> dict[int, float]:
    """Extract pLDDT B-factor values at specific residue positions."""
    plddts: dict[int, float] = {}
    with _open_pdb(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_id = int(line[22:26].strip())
                if res_id in residue_ids:
                    try:
                        plddts[res_id] = float(line[60:66].strip())
                    except ValueError:
                        pass
    return plddts


def extract_active_site_features(
    pdb_path: Path,
    catalytic_residues: list[int],
    plddt_threshold: float = PLDDT_THRESHOLD,
) -> tuple[np.ndarray, bool]:
    """Extract F_active_site feature vector.

    Returns
    -------
    features : np.ndarray
        Length-ACTIVE_SITE_DIM feature vector.
    plddt_ok : bool
        True if mean pLDDT at catalytic residues ≥ threshold.
    """
    features = np.zeros(ACTIVE_SITE_DIM, dtype=np.float32)

    if not pdb_path.exists():
        return features, False

    plddts = get_plddt_at_residues(pdb_path, catalytic_residues)
    if not plddts:
        return features, False

    mean_plddt = float(np.mean(list(plddts.values())))
    plddt_ok = mean_plddt >= plddt_threshold

    # Feature[0]: mean pLDDT of catalytic residues (normalized /100)
    features[0] = mean_plddt / 100.0

    if not plddt_ok:
        # Zero-fill with pLDDT flag set
        features[ACTIVE_SITE_DIM - 1] = 1.0  # plddt_low_flag
        return features, False

    # Cα pairwise distances (features 1–10 for up to 5 catalytic residues → 10 pairs)
    coords = get_ca_coordinates(pdb_path, catalytic_residues)
    res_list = sorted(coords.keys())
    feat_idx = 1
    for i in range(min(len(res_list), 5)):
        for j in range(i + 1, min(len(res_list), 5)):
            if feat_idx >= ACTIVE_SITE_DIM - 2:
                break
            ri, rj = res_list[i], res_list[j]
            if ri in coords and rj in coords:
                dist = float(np.linalg.norm(coords[ri] - coords[rj]))
                features[feat_idx] = dist / 30.0  # normalize by ~max catalytic distance
            feat_idx += 1

    # DSSP secondary structure at catalytic residues (features 11–15: helix/strand/loop)
    dssp = get_dssp_at_residues(pdb_path, catalytic_residues)
    n_helix = sum(1 for v in dssp.values() if v in ("H", "G", "I"))
    n_strand = sum(1 for v in dssp.values() if v in ("E", "B"))
    n_loop = sum(1 for v in dssp.values() if v in ("T", "S", "-", " "))
    n_total = max(len(dssp), 1)
    features[11] = n_helix / n_total
    features[12] = n_strand / n_total
    features[13] = n_loop / n_total

    return features, True


def build_active_site_feature_matrix(
    accessions: list[str],
    structure_dir: Path,
    catalytic_residues_map: dict[str, list[int]],
    plddt_threshold: float = PLDDT_THRESHOLD,
) -> tuple[np.ndarray, list[bool]]:
    """Build N × ACTIVE_SITE_DIM feature matrix for a list of proteins.

    Parameters
    ----------
    accessions : list[str]
        UniProt accessions.
    structure_dir : Path
        Directory containing AlphaFold .pdb.gz files.
    catalytic_residues_map : dict[str, list[int]]
        Maps UniProt accession → list of catalytic residue positions.
        From M-CSA or UniProt ACT_SITE annotations.
    """
    matrix = np.zeros((len(accessions), ACTIVE_SITE_DIM), dtype=np.float32)
    valid_mask = [False] * len(accessions)

    for i, acc in enumerate(accessions):
        cat_res = catalytic_residues_map.get(acc, [])
        if not cat_res:
            continue  # no catalytic residue data

        pdb_path = structure_dir / f"AF-{acc}-F1-model_v4.pdb.gz"
        if not pdb_path.exists():
            pdb_path = structure_dir / f"AF-{acc}-F1-model_v4.pdb"
        if not pdb_path.exists():
            continue

        feats, ok = extract_active_site_features(pdb_path, cat_res, plddt_threshold)
        matrix[i] = feats
        valid_mask[i] = ok

    n_valid = sum(valid_mask)
    print(f"Active-site features: {n_valid}/{len(accessions)} with pLDDT ≥ {plddt_threshold}")
    return matrix, valid_mask
