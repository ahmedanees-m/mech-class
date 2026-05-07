"""Step 11 — Compute active-site geometry features F_active_site (7-dim) (Week 4).

Pfam-guided catalytic-residue detection without per-protein M-CSA annotations.

Algorithm per protein:
  1. Load present Pfam families from F_domain.parquet (dom_0..dom_22 columns).
  2. Look up expected catalytic residue amino acid types from
     mech_class/data/active_site_residues.yaml using the highest-priority
     matching Pfam family.
  3. In the AlphaFold PDB, extract CA coordinates + pLDDT for all residues of
     those types.
  4. Select the tightest spatial cluster of n_select residues via O(n²) greedy
     nearest-neighbour: seed on the residue whose nearest-neighbour pair has
     the minimum distance, then expand greedily by centroid proximity.
  5. Compute 7-dim geometry features from the selected cluster.

Features (unchanged from original plan):
  F0: catalytic residue count normalised by 10
  F1: mean pairwise CA distance (Å) normalised by 30
  F2: std  pairwise CA distance (Å) normalised by 10
  F3: mean pLDDT at selected residues normalised by 100
  F4: std  pLDDT at selected residues normalised by 30
  F5: radius of gyration of CA atoms normalised by 30
  F6: has_active_site_annotation (1.0 if cluster found, else 0.0)

Proteins with no AF structure or fewer than min_count candidates are zero-filled.

Run via:
    docker run --rm \\
        -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "
            SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps;
            pip install -e . --quiet;
            python scripts/12_compute_active_site_features.py"

Expected output:
  /data/features/active_site/F_active_site.parquet
    columns: uniprot_acc, as_0..as_6, zero_filled, has_F_active_site
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

LABELS    = Path("/data/labels/mechanism_labels_final.parquet")
AF_CACHE  = Path("/data/structures/alphafold")
F_DOMAIN  = Path("/data/features/domain/F_domain.parquet")
OUT       = Path("/data/features/active_site/F_active_site.parquet")
YAML_PATH = Path("/pkg/mech_class/data/active_site_residues.yaml")

FEAT_DIM = 7
ACC_COL  = "uniprot_acc"

OUT.parent.mkdir(parents=True, exist_ok=True)

# Pfam whitelist order (matches dom_0..dom_22 columns in F_domain.parquet)
WHITELIST_ORDER = [
    "PF13395", "PF18541", "PF16595", "PF18516",
    "PF01548", "PF02371", "PF07282",
    "PF00665", "PF01609", "PF13586",
    "PF08721", "PF11426", "PF05621",
    "PF00589", "PF00239", "PF07508",
    "PF01844", "PF02486",
    # auxiliary (dom_18..dom_22) — not used for active-site lookup
    "PF18061", "PF16592", "PF16593", "PF13639", "PF03377",
]


# ── YAML loader ──────────────────────────────────────────────────────────────

def _load_yaml() -> list[dict]:
    """Return active_site_residues.yaml entries list."""
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"active_site_residues.yaml not found at {YAML_PATH}")
    raw = yaml.safe_load(YAML_PATH.read_text())
    return raw["entries"]


# ── Pfam presence from F_domain ───────────────────────────────────────────────

def _get_present_pfams(row: pd.Series) -> list[str]:
    """Return list of Pfam accessions present (dom_X > 0.5) for one protein."""
    present = []
    for i, pf in enumerate(WHITELIST_ORDER):
        col = f"dom_{i}"
        if col in row.index and row[col] > 0.5:
            present.append(pf)
    return present


# ── PDB parsing ──────────────────────────────────────────────────────────────

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _parse_pdb(pdb_path: Path) -> dict[int, dict]:
    """Parse CA atoms from PDB.  Returns {resnum: {aa, x, y, z, plddt}}."""
    result: dict[int, dict] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") and line[12:16].strip() == "CA"):
                continue
            try:
                resnum = int(line[22:26].strip())
                aa3    = line[17:20].strip()
                aa1    = _AA3_TO_1.get(aa3, "X")
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                plddt  = float(line[60:66].strip())
                result[resnum] = {"aa": aa1, "xyz": np.array([x, y, z], np.float32),
                                  "plddt": plddt}
            except ValueError:
                pass
    return result


# ── Cluster selection ─────────────────────────────────────────────────────────

def _select_cluster(candidates: list[tuple[int, np.ndarray, float]],
                    n: int, use_cluster: bool) -> list[tuple[int, np.ndarray, float]]:
    """Select n candidates forming the tightest spatial cluster.

    Parameters
    ----------
    candidates : list of (resnum, xyz, plddt)
    n          : number of residues to select
    use_cluster: if False, return first n (N-terminal bias)

    Returns
    -------
    list of up to n (resnum, xyz, plddt) tuples
    """
    if len(candidates) <= n:
        return candidates
    if not use_cluster:
        return candidates[:n]
    if n == 1:
        # Return candidate with highest pLDDT
        return [max(candidates, key=lambda c: c[2])]

    coords = np.array([c[1] for c in candidates])  # (N, 3)
    N = len(candidates)

    # Greedy expansion: seed on pair with minimum distance, then expand by centroid
    # Step 1: find seed pair
    best_d   = float("inf")
    seed_i, seed_j = 0, 1
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < best_d:
                best_d, seed_i, seed_j = d, i, j

    selected_idx = {seed_i, seed_j}

    # Step 2: greedily add closest to current centroid
    while len(selected_idx) < n:
        centroid = coords[list(selected_idx)].mean(axis=0)
        best_d   = float("inf")
        best_k   = -1
        for k in range(N):
            if k in selected_idx:
                continue
            d = float(np.linalg.norm(coords[k] - centroid))
            if d < best_d:
                best_d, best_k = d, k
        selected_idx.add(best_k)

    return [candidates[i] for i in sorted(selected_idx)]


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_features(selected: list[tuple[int, np.ndarray, float]]) -> np.ndarray:
    """Compute 7-dim active-site geometry features from selected CA atoms."""
    vec = np.zeros(FEAT_DIM, dtype=np.float32)
    n   = len(selected)
    if n == 0:
        return vec

    coords = np.array([s[1] for s in selected], dtype=np.float32)
    plddts = np.array([s[2] for s in selected], dtype=np.float32)

    vec[0] = n / 10.0

    if n >= 2:
        dists = [float(np.linalg.norm(coords[i] - coords[j]))
                 for i in range(n) for j in range(i + 1, n)]
        vec[1] = np.mean(dists) / 30.0
        vec[2] = np.std(dists)  / 10.0
        centroid = coords.mean(axis=0)
        vec[5] = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))) / 30.0

    vec[3] = plddts.mean() / 100.0
    vec[4] = plddts.std()  / 30.0
    vec[6] = 1.0            # has_active_site_annotation
    return vec


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    if not LABELS.exists():
        print(f"Labels not found at {LABELS}.")
        return
    if not F_DOMAIN.exists():
        print(f"F_domain not found at {F_DOMAIN}. Run script 13 first.")
        return

    labels  = pd.read_parquet(LABELS)
    fdomain = pd.read_parquet(F_DOMAIN).set_index(ACC_COL)
    accs    = labels[ACC_COL].tolist()
    entries = _load_yaml()
    print(f"Processing {len(accs)} proteins for active-site features")
    print(f"YAML entries loaded: {len(entries)}")

    # Build Pfam → YAML entry lookup (priority = list order)
    pfam_to_entry: dict[str, dict] = {}
    for entry in entries:
        pf = entry["pfam"]
        if pf not in pfam_to_entry:
            pfam_to_entry[pf] = entry

    records: list[dict] = []
    n_annotated  = 0
    n_zero       = 0
    pfam_miss    = 0
    struct_miss  = 0

    for i, acc in enumerate(accs, 1):
        if i % 50 == 0 or i == 1:
            print(f"  {i}/{len(accs)}")

        zero_filled = False
        has_anno    = False
        vec         = np.zeros(FEAT_DIM, dtype=np.float32)

        # 1. Find AF structure (any cached version)
        pdb_path: Optional[Path] = next(
            (AF_CACHE / f"AF-{acc}-F1-model_v{v}.pdb"
             for v in [6, 5, 4, 3, 2]
             if (AF_CACHE / f"AF-{acc}-F1-model_v{v}.pdb").exists()),
            None,
        )
        if pdb_path is None:
            zero_filled = True
            struct_miss += 1
        else:
            # 2. Identify present Pfam families
            if acc in fdomain.index:
                row = fdomain.loc[acc]
                present_pfams = _get_present_pfams(row)
            else:
                present_pfams = []

            # 3. Find highest-priority YAML entry matching a present Pfam
            matched_entry = None
            for entry in entries:         # already priority-ordered
                if entry["pfam"] in present_pfams:
                    matched_entry = entry
                    break

            if matched_entry is None:
                zero_filled = True
                pfam_miss  += 1
            else:
                # 4. Parse PDB and collect candidates
                try:
                    parsed    = _parse_pdb(pdb_path)
                    cat_types = set(matched_entry["catalytic_types"])
                    n_select  = int(matched_entry["n_select"])
                    min_count = int(matched_entry.get("min_count", 1))
                    use_clust = bool(matched_entry.get("cluster", True))

                    candidates = [
                        (rn, info["xyz"], info["plddt"])
                        for rn, info in sorted(parsed.items())
                        if info["aa"] in cat_types
                    ]

                    if len(candidates) < min_count:
                        zero_filled = True
                        pfam_miss  += 1
                    else:
                        selected = _select_cluster(candidates, n_select, use_clust)
                        vec      = _compute_features(selected)
                        has_anno = True
                        n_annotated += 1

                except Exception as exc:
                    print(f"  [WARN] {acc}: {exc}")
                    zero_filled = True

        if zero_filled:
            n_zero += 1

        rec = {ACC_COL: acc, "zero_filled": zero_filled,
               "has_F_active_site": has_anno}
        for j, v in enumerate(vec):
            rec[f"as_{j}"] = float(v)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(OUT, compression="zstd")

    print(f"\n=== Active-site features complete ===")
    print(f"Proteins:            {len(df)}")
    print(f"Annotated (cluster): {n_annotated}  ({n_annotated/len(df)*100:.1f}%)")
    print(f"Zero-filled:         {n_zero}  ({n_zero/len(df)*100:.1f}%)")
    print(f"  No AF structure:   {struct_miss}")
    print(f"  No YAML Pfam match:{pfam_miss}")
    print(f"Output -> {OUT}")


if __name__ == "__main__":
    run()
