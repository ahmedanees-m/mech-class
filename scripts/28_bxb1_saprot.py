"""Step 18b -- SaProt features for Bxb1 (O25753) + Tier-A re-prediction.

Downloads AlphaFold structure for O25753, runs Foldseek 3Di tokenisation,
infers SaProt 650M embedding, then re-runs Tier-A prediction with the
full feature vector (ESM-2 seq + SaProt struct + canonical domain flags).

Answers: does adding struct features fix the Bxb1 DSB_FREE/DSB_NUCLEASE
misclassification?

Three possible outcomes:
  (a) Tier-A → DSB_FREE_TRANSEST_RECOMBINASE   struct fixes it
  (b) Tier-A → DSB_NUCLEASE (still)             document as model limitation
  (c) No AF structure / pLDDT < 70              struct unavailable → (b)

Input (already in /data):
  /data/features/fused/feature_matrix.parquet   (for ESM-2 seq row)
  /data/models/lgbm_tier_a/model.pkl

Run via:
    docker run --rm --gpus all \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "git config --global --add safe.directory /pkg && \\
                 git config --global --add safe.directory /genome-atlas && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps && \\
                 pip install lightgbm scikit-learn requests --quiet && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --quiet && \\
                 python scripts/28_bxb1_saprot.py"

Output:
  /data/validation/bxb1_saprot_result.json
"""
from __future__ import annotations
import json
import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

DATA_DIR    = Path("/data")
FEAT_PATH         = Path("/data/features/fused/feature_matrix.parquet")
LABEL_PATH        = Path("/data/features/fused/labels.parquet")
HOLDOUT_FEAT_PATH = Path("/data/validation/holdout_features.parquet")
MODEL_DIR   = Path("/data/models")
TIER_A_PATH = MODEL_DIR / "tier_a" / "model.pkl"
OUT_PATH    = Path("/data/validation/bxb1_saprot_result.json")

BXB1_ACC    = "O25753"
BXB1_NAME   = "Bxb1_integrase"
AF_URL      = "https://alphafold.ebi.ac.uk/files/AF-O25753-F1-model_v4.pdb"
AF_URL_V3   = "https://alphafold.ebi.ac.uk/files/AF-O25753-F1-model_v3.pdb"

PFAM_WHITELIST = [
    "PF13395","PF18541","PF16595","PF18516","PF01548","PF02371","PF07282",
    "PF00665","PF01609","PF13586","PF08721","PF11426","PF05621","PF00589",
    "PF00239","PF07508","PF01844","PF02486","PF18061","PF16592","PF16593",
    "PF13639","PF03377",
]
CANONICAL_PFAM_BXB1 = ["PF07508"]   # Large serine integrase catalytic domain
CLASSES = ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE", "TRANSPOSASE"]

PLDDT_THRESHOLD = 70.0


# ── AlphaFold download ────────────────────────────────────────────────────────

def download_af_structure(out_pdb: Path) -> bool:
    """Download AlphaFold PDB for O25753. Returns True if succeeded."""
    for url in [AF_URL, AF_URL_V3]:
        try:
            print(f"  Trying {url} ...")
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                out_pdb.write_bytes(r.content)
                print(f"  Downloaded {out_pdb} ({len(r.content)//1024} KB)")
                return True
            else:
                print(f"  HTTP {r.status_code}")
        except Exception as e:
            print(f"  [WARN] {e}")
    return False


def parse_plddt(pdb_path: Path) -> float:
    """Mean pLDDT from ATOM B-factors in PDB."""
    scores = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    scores.append(float(line[60:66].strip()))
                except ValueError:
                    pass
    return float(np.mean(scores)) if scores else 0.0


# ── Foldseek 3Di tokens ──────────────────────────────────────────────────────

def run_foldseek_3di(pdb_path: Path, workdir: Path) -> str | None:
    """
    Run Foldseek to produce 3Di sequence string for a single PDB file.
    Returns the 3Di token string, or None on failure.
    """
    db_dir    = workdir / "fsdb"
    db_dir.mkdir(exist_ok=True)
    fasta_out = workdir / "bxb1_3di.fasta"

    # createdb
    cmd_create = [
        "foldseek", "createdb", str(pdb_path), str(db_dir / "db"),
        "--threads", "1",
    ]
    # structureto3didescriptor (produces .ss3 with 3Di tokens)
    cmd_ss3 = [
        "foldseek", "structureto3didescriptor",
        str(db_dir / "db"), str(workdir / "bxb1"), "--threads", "1",
    ]
    # convert2fasta
    cmd_fa = [
        "foldseek", "convert2fasta",
        str(db_dir / "db_ss"), str(fasta_out),
    ]

    for cmd in [cmd_create, cmd_ss3, cmd_fa]:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [WARN] Foldseek exit {result.returncode}: {result.stderr[:300]}")

    # Try to read the 3Di FASTA
    if fasta_out.exists():
        lines = fasta_out.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.startswith(">") and line.strip():
                return line.strip().lower()

    # Alternative: try lndb → convert2fasta approach
    return _foldseek_alt(pdb_path, workdir)


def _foldseek_alt(pdb_path: Path, workdir: Path) -> str | None:
    """Alternative Foldseek approach: single-file direct descriptor."""
    out_tsv = workdir / "bxb1_3di.tsv"
    cmd = [
        "foldseek", "easy-search", str(pdb_path), str(pdb_path),
        str(workdir / "result"), str(workdir / "tmp"),
        "--format-output", "query,qss",
        "--threads", "1",
        "-v", "0",
    ]
    print(f"  Alt Foldseek: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    res_file = workdir / "result"
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                seq = parts[1].strip().lower()
                print(f"  3Di sequence (alt): length={len(seq)}")
                return seq

    print(f"  [WARN] Could not extract 3Di tokens. Foldseek stderr: {r.stderr[:200]}")
    return None


# ── SaProt inference ─────────────────────────────────────────────────────────

def saprot_embed(
    aa_seq: str,
    di3_seq: str | None,
    model_name: str = "westlake-repl/SaProt_650M_AF2",
) -> np.ndarray | None:
    """
    Run SaProt 650M to get 1280-dim struct embedding.
    If di3_seq is None, returns None (zero-fill caller).
    Interleaves amino-acid and 3Di tokens: A#v#L#...
    """
    try:
        from transformers import EsmTokenizer, EsmModel
    except ImportError:
        print("  [ERROR] transformers not available")
        return None

    if di3_seq is None:
        print("  [WARN] No 3Di tokens — cannot run SaProt")
        return None

    # Interleave: each AA gets its paired 3Di token
    # SaProt expects input like "M#a#K#b#..." (AA + '#' + 3Di_token)
    if len(aa_seq) != len(di3_seq):
        # Truncate to shorter
        min_len = min(len(aa_seq), len(di3_seq))
        aa_seq  = aa_seq[:min_len]
        di3_seq = di3_seq[:min_len]
        print(f"  [WARN] AA/3Di length mismatch, truncated to {min_len}")

    saprot_input = "".join(f"{a}#{d}" for a, d in zip(aa_seq, di3_seq))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading SaProt from {model_name} on {device}")
    print(f"  Input length: {len(aa_seq)} residues")

    try:
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model     = EsmModel.from_pretrained(model_name).to(device).eval()
    except Exception as e:
        print(f"  [ERROR] SaProt load failed: {e}")
        return None

    inputs = tokenizer(saprot_input, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    # Mean-pool last hidden state over residue positions (exclude BOS/EOS)
    hidden = out.last_hidden_state[0, 1:-1, :]   # [L, 1280]
    emb    = hidden.mean(dim=0).cpu().float().numpy()  # [1280]
    print(f"  SaProt embedding: shape={emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}")
    return emb


# ── Feature assembly ──────────────────────────────────────────────────────────

def build_feature_vector(
    esm2_vec   : np.ndarray,   # [640]
    saprot_vec : np.ndarray,   # [1280] or None
    pfam_list  : list[str],
) -> np.ndarray:
    """Assemble [640 + 1280 + 26 + 7] = 1953-dim feature vector."""
    seq_feat    = esm2_vec.astype(np.float32)

    if saprot_vec is not None:
        struct_feat = saprot_vec.astype(np.float32)
    else:
        struct_feat = np.zeros(1280, dtype=np.float32)
        print("  struct_* = zeros (SaProt unavailable)")

    # Domain features (26 dims)
    dom_feat = np.zeros(26, dtype=np.float32)
    for pf in pfam_list:
        if pf in PFAM_WHITELIST:
            dom_feat[PFAM_WHITELIST.index(pf)] = 1.0
    # dom_23: IS110 composite (PF01548 AND PF02371)
    dom_feat[23] = float("PF01548" in pfam_list and "PF02371" in pfam_list)
    # dom_24: editor fusion (no composite here)
    dom_feat[24] = 0.0
    # dom_25: single-domain (exactly 1 Pfam hit)
    hit_count = sum(1 for p in pfam_list if p in PFAM_WHITELIST)
    dom_feat[25] = float(hit_count == 1)

    # Active-site geometry (7 dims) — zero-filled
    as_feat = np.zeros(7, dtype=np.float32)

    feat = np.concatenate([seq_feat, struct_feat, dom_feat, as_feat])
    return feat


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    result = {
        "accession":       BXB1_ACC,
        "name":            BXB1_NAME,
        "struct_source":   "zero",
        "mean_plddt":      None,
        "tier_a_pred":     None,
        "tier_a_conf":     None,
        "tier_a_proba":    None,
        "fixed_by_struct": None,
        "notes":           [],
    }

    # ── Load Tier-A model ─────────────────────────────────────────────────
    print(f"Loading Tier-A model from {TIER_A_PATH}")
    with open(TIER_A_PATH, "rb") as fh:
        tier_a_bundle = pickle.load(fh)
    lgbm        = tier_a_bundle["model"]
    feat_cols   = tier_a_bundle["feature_cols"]
    le          = tier_a_bundle["label_encoder"]
    print(f"  Feature dims: {len(feat_cols)}")

    # ── Get ESM-2 row for O25753 ───────────────────────────────────────────
    # O25753 is in HOLDOUT_SET — excluded from training matrix.
    # Load from holdout_features.parquet (computed by script 26).
    print(f"\nLoading ESM-2 features for {BXB1_ACC} ...")
    bxb1_row = None
    for feat_src in [HOLDOUT_FEAT_PATH, FEAT_PATH]:
        if not feat_src.exists():
            continue
        fd = pd.read_parquet(feat_src)
        acc_col = "uniprot_acc" if "uniprot_acc" in fd.columns else "accession"
        match = fd[fd[acc_col] == BXB1_ACC]
        if len(match) > 0:
            bxb1_row = match
            print(f"  Found in {feat_src} ({len(match)} row)")
            break
    if bxb1_row is None or len(bxb1_row) == 0:
        print(f"  [ERROR] {BXB1_ACC} not found in any feature parquet")
        result["notes"].append(f"{BXB1_ACC} not found in holdout or training features")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(result, indent=2))
        return

    seq_cols = [f"seq_{i}" for i in range(640)]
    esm2_vec = bxb1_row[seq_cols].values[0].astype(np.float32)
    print(f"  ESM-2 vec: shape={esm2_vec.shape}, mean={esm2_vec.mean():.4f}")

    # Also get AA sequence from the row (if stored) — else fetch from UniProt
    aa_seq = None
    if "sequence" in bxb1_row.columns:
        aa_seq = str(bxb1_row["sequence"].iloc[0])
    else:
        try:
            r = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{BXB1_ACC}.fasta",
                timeout=30
            )
            if r.status_code == 200:
                lines = r.text.splitlines()
                aa_seq = "".join(l for l in lines if not l.startswith(">"))
            print(f"  Fetched sequence: {len(aa_seq)} AA")
        except Exception as e:
            print(f"  [WARN] Could not fetch sequence: {e}")

    # ── Baseline: predict with zero struct ────────────────────────────────
    print("\n--- Baseline prediction (struct=zeros, same as script 26) ---")
    feat_zero = build_feature_vector(esm2_vec, None, CANONICAL_PFAM_BXB1)
    pred_zero, conf_zero, proba_zero = predict(lgbm, le, feat_cols, feat_zero)
    print(f"  Tier-A (zero struct): {pred_zero}  conf={conf_zero:.4f}  proba={proba_zero}")

    # ── Download AlphaFold structure ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_path = tmpdir / f"AF-{BXB1_ACC}-F1.pdb"

        print(f"\nDownloading AlphaFold structure for {BXB1_ACC} ...")
        af_ok = download_af_structure(pdb_path)

        if not af_ok:
            print("  [WARN] AlphaFold structure unavailable — using zero struct")
            result["struct_source"] = "zero (AF download failed)"
            result["notes"].append("AlphaFold structure download failed")
        else:
            mean_plddt = parse_plddt(pdb_path)
            result["mean_plddt"] = round(mean_plddt, 2)
            print(f"  Mean pLDDT: {mean_plddt:.1f}")

            if mean_plddt < PLDDT_THRESHOLD:
                print(f"  [WARN] pLDDT {mean_plddt:.1f} < {PLDDT_THRESHOLD} — gating struct to zero")
                result["struct_source"] = f"zero (pLDDT={mean_plddt:.1f} < {PLDDT_THRESHOLD})"
                result["notes"].append(f"Low pLDDT ({mean_plddt:.1f}) — struct zero-filled per training protocol")
                saprot_vec = None
            else:
                print(f"  pLDDT OK ({mean_plddt:.1f} >= {PLDDT_THRESHOLD}) — running Foldseek + SaProt")
                result["struct_source"] = f"SaProt_650M_AF2 (pLDDT={mean_plddt:.1f})"

                # Run Foldseek
                print("\nRunning Foldseek 3Di tokenisation ...")
                di3_seq = run_foldseek_3di(pdb_path, tmpdir)
                if di3_seq:
                    print(f"  3Di sequence length: {len(di3_seq)}")
                else:
                    print("  [WARN] Foldseek failed — no 3Di tokens")
                    result["notes"].append("Foldseek 3Di tokenisation failed")

                # Run SaProt
                print("\nRunning SaProt 650M inference ...")
                saprot_vec = saprot_embed(aa_seq, di3_seq) if aa_seq else None

                if saprot_vec is None:
                    result["struct_source"] = "zero (SaProt inference failed)"
                    result["notes"].append("SaProt inference failed — struct zero-filled")

            # ── Full-feature prediction ───────────────────────────────────
            print("\n--- Full-feature prediction ---")
            feat_full = build_feature_vector(esm2_vec, saprot_vec, CANONICAL_PFAM_BXB1)
            pred_full, conf_full, proba_full = predict(lgbm, le, feat_cols, feat_full)
            print(f"  Tier-A (full struct): {pred_full}  conf={conf_full:.4f}  proba={proba_full}")

            result["tier_a_pred"]     = pred_full
            result["tier_a_conf"]     = round(conf_full, 4)
            result["tier_a_proba"]    = {CLASSES[i]: round(float(proba_full[i]), 4) for i in range(3)}
            result["fixed_by_struct"] = (pred_full == "DSB_FREE_TRANSEST_RECOMBINASE")

            # Summarise
            print("\n=== Bxb1 SaProt Result ===")
            print(f"  Baseline (zero struct): {pred_zero}  conf={conf_zero:.4f}")
            print(f"  Full struct:            {pred_full}  conf={conf_full:.4f}")
            if result["fixed_by_struct"]:
                print("  ✓ STRUCT FIXES MISCLASSIFICATION — DSB_FREE_TRANSEST_RECOMBINASE")
            else:
                print("  ✗ STILL MISCLASSIFIED — document as model limitation")
                print("    Root cause: ESM-2 embeddings dominate; Bxb1 serine-integrase")
                print("    chemistry overlaps with nuclease fold at sequence level.")

    result["baseline_pred"] = pred_zero
    result["baseline_conf"] = round(conf_zero, 4)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved -> {OUT_PATH}")


def predict(lgbm, le, feat_cols, feat_vec):
    """Run Tier-A prediction for a single feature vector."""
    full_names = (
        [f"seq_{i}" for i in range(640)]
        + [f"struct_{i}" for i in range(1280)]
        + [f"dom_{i}" for i in range(26)]
        + [f"as_{i}" for i in range(7)]
    )
    row = dict(zip(full_names, feat_vec.tolist()))
    df  = pd.DataFrame([row])
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feat_cols].values.astype(np.float32)

    proba = lgbm.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    pred  = le.classes_[idx]
    return pred, float(proba[idx]), proba


if __name__ == "__main__":
    run()
