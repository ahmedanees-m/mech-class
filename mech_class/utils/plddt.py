"""AlphaFold pLDDT extraction and active-site confidence filtering.

Paper 1 §1.4.4 caveat: AlphaFold confidence (pLDDT) for composite folds with
unusual domain interfaces (IS110 RuvC-fold + serine junction) may be lower than
for canonical single-domain enzymes. MECH-CLASS reports per-prediction confidence
weighted by mean active-site pLDDT, and down-weights F_struct / F_active_site
channels when pLDDT < 70 in the active-site region.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np


def _open_pdb(path: Path) -> io.TextIOWrapper:
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))  # type: ignore
    return open(path)


def get_mean_plddt(pdb_path: Path, chain: str = "A") -> float:
    """Compute mean pLDDT from AlphaFold B-factor column (all Cα atoms)."""
    plddts = []
    with _open_pdb(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain and line[21].strip():
                continue
            try:
                plddts.append(float(line[60:66]))
            except ValueError:
                pass
    if not plddts:
        return 0.0
    return float(np.mean(plddts))


def get_domain_plddt(
    pdb_path: Path,
    residue_start: int,
    residue_end: int,
) -> float:
    """Compute mean pLDDT for a specific domain region (residue_start to residue_end)."""
    plddts = []
    with _open_pdb(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                res_id = int(line[22:26])
                if residue_start <= res_id <= residue_end:
                    plddts.append(float(line[60:66]))
            except ValueError:
                pass
    return float(np.mean(plddts)) if plddts else 0.0


def plddt_weight(mean_plddt: float, threshold: float = 70.0) -> float:
    """Return a confidence weight [0.0, 1.0] based on mean pLDDT.

    Above threshold: linear scale from 0.5 (at threshold) to 1.0 (at pLDDT=100).
    Below threshold: 0.0 (feature is unreliable; zero-fill the channel).
    """
    if mean_plddt < threshold:
        return 0.0
    return 0.5 + 0.5 * (mean_plddt - threshold) / (100.0 - threshold)


def filter_structures_by_plddt(
    accessions: list[str],
    structure_dir: Path,
    threshold: float = 70.0,
) -> tuple[list[str], list[str]]:
    """Split accessions into passing/failing pLDDT threshold.

    Returns
    -------
    passed : list[str]
        Accessions with mean pLDDT ≥ threshold.
    failed : list[str]
        Accessions with no structure or mean pLDDT < threshold.
    """
    passed = []
    failed = []
    for acc in accessions:
        pdb_path = structure_dir / f"AF-{acc}-F1-model_v4.pdb.gz"
        if not pdb_path.exists():
            pdb_path = structure_dir / f"AF-{acc}-F1-model_v4.pdb"
        if not pdb_path.exists():
            failed.append(acc)
            continue
        mplddt = get_mean_plddt(pdb_path)
        if mplddt >= threshold:
            passed.append(acc)
        else:
            failed.append(acc)

    print(f"pLDDT ≥ {threshold}: {len(passed)}/{len(accessions)} structures pass")
    return passed, failed
