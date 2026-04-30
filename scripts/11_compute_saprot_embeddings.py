"""Compute SaProt 650M structure-aware embeddings.

SaProt (Su et al. 2023, westlake-realbio/SaProt_650M_AF2) encodes each protein as
interleaved AA + Foldseek 3Di tokens ("MaDrKg...") and produces a 1280-dim
mean-pooled embedding.

Pipeline per protein:
  1. Download AlphaFold2 PDB (v4) from EMBL-EBI if not cached.
  2. Parse mean pLDDT from B-factor column.
  3. Get AA sequence from ATLAS DuckDB nodes_protein.
  4. Run Foldseek createdb + convert2fasta on the _ss db to get 3Di tokens.
  5. Interleave: "".join(aa+di for aa,di in zip(seq, tokens_3di)).
  6. Tokenize + run SaProt forward pass; mean-pool last hidden state.
  7. pLDDT gate: mean pLDDT < 70 -> zero-fill embedding.
  8. After all proteins: run collapse audit (>= 95% unique rows required).

Run via:
    docker run --rm --gpus all \\
        -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "
            SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps;
            pip install -e . --quiet;
            python scripts/11_compute_saprot_embeddings.py"

Expected output:
  /data/features/struct/F_struct.parquet
    columns: uniprot_acc, struct_0 .. struct_1279, plddt_mean, zero_filled
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

LABELS     = Path("/data/labels/mechanism_labels_final.parquet")
ATLAS_DB   = Path("/data/graphs/atlas.duckdb")
AF_CACHE   = Path("/data/structures/alphafold")
OUT        = Path("/data/features/struct/F_struct.parquet")

SAPROT_MODEL   = "westlake-repl/SaProt_650M_AF2"
EMBED_DIM      = 1280
PLDDT_THRESH   = 70.0
MAX_SEQ_LEN    = 1022          # SaProt effective max (1024 - CLS/EOS)
ACC_COL        = "uniprot_acc"

AF_CACHE.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)


# --- AlphaFold structure download --------------------------------------------

def _fetch_af_pdb(acc: str) -> Path | None:
    """Download AF2 PDB for accession (tries v6->v2); return path or None on failure."""
    # Return any already-cached version
    for ver in [6, 5, 4, 3, 2]:
        cached = AF_CACHE / f"AF-{acc}-F1-model_v{ver}.pdb"
        if cached.exists():
            return cached
    # Try to download from latest version down
    for ver in [6, 5, 4, 3, 2]:
        pdb_path = AF_CACHE / f"AF-{acc}-F1-model_v{ver}.pdb"
        url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"
        try:
            urllib.request.urlretrieve(url, pdb_path)
            return pdb_path
        except Exception:
            continue
    print(f"    [WARN] AF download failed for {acc}: no version found (v2-v6)")
    return None


# --- pLDDT parsing -----------------------------------------------------------

def _mean_plddt(pdb_path: Path) -> float:
    """Parse per-residue pLDDT (B-factor) from PDB ATOM records (CA only)."""
    vals: list[float] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    vals.append(float(line[60:66]))
                except ValueError:
                    pass
    return float(np.mean(vals)) if vals else 0.0


# --- AA sequence from ATLAS --------------------------------------------------

def _load_sequences(accessions: list[str]) -> dict[str, str]:
    """Return {accession: AA sequence} from ATLAS DuckDB nodes_protein."""
    acc_sql = ", ".join(f"'{a}'" for a in accessions)
    con = duckdb.connect(str(ATLAS_DB), read_only=True)
    df = con.execute(
        f"SELECT accession, sequence FROM nodes_protein WHERE accession IN ({acc_sql})"
    ).fetchdf()
    con.close()
    return dict(zip(df["accession"], df["sequence"]))


# --- Foldseek 3Di tokens -----------------------------------------------------

def _foldseek_3di(pdb_path: Path) -> str | None:
    """Run Foldseek and return the 3Di structural-alphabet token string."""
    with tempfile.TemporaryDirectory() as tmp:
        db  = os.path.join(tmp, "db")
        out = os.path.join(tmp, "3di.fasta")
        r1 = subprocess.run(
            ["foldseek", "createdb", str(pdb_path), db, "--threads", "1"],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            return None
        # Link the sequence header DB to the _ss DB so convert2fasta can find it
        subprocess.run(
            ["foldseek", "lndb", db + "_h", db + "_ss_h"],
            capture_output=True, text=True,
        )
        r2 = subprocess.run(
            ["foldseek", "convert2fasta", db + "_ss", out],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            return None
        with open(out) as fh:
            lines = fh.readlines()
        tokens = "".join(l.strip() for l in lines if not l.startswith(">"))
        # SaProt expects lowercase 3Di tokens (e.g. "MmAkTp..."); Foldseek emits uppercase
        return tokens.lower() if tokens else None


# --- SaProt inference --------------------------------------------------------

def _load_saprot(device: str):
    from transformers import EsmTokenizer, EsmModel
    tok   = EsmTokenizer.from_pretrained(SAPROT_MODEL)
    model = EsmModel.from_pretrained(SAPROT_MODEL).to(device).eval()
    return tok, model


def _embed(seq_aa: str, tokens_3di: str, tokenizer, model, device: str) -> np.ndarray:
    """Return 1280-dim mean-pooled SaProt embedding."""
    # Align lengths (truncate to shorter; should be equal from same AF PDB)
    n = min(len(seq_aa), len(tokens_3di), MAX_SEQ_LEN)
    interleaved = "".join(a + d for a, d in zip(seq_aa[:n], tokens_3di[:n]))
    inputs = tokenizer(interleaved, return_tensors="pt",
                       truncation=True, max_length=MAX_SEQ_LEN + 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # Mean-pool over residue positions (exclude CLS at position 0)
    hidden = out.last_hidden_state[0, 1:, :]   # (seq_len, 1280)
    return hidden.mean(dim=0).cpu().numpy().astype(np.float32)


# --- Main --------------------------------------------------------------------

def run() -> None:
    if not LABELS.exists():
        print(f"Labels not found at {LABELS}.")
        return

    labels     = pd.read_parquet(LABELS)
    accessions = labels[ACC_COL].tolist()
    print(f"Processing {len(accessions)} proteins with SaProt {SAPROT_MODEL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading SaProt model...")
    tokenizer, model = _load_saprot(device)

    print("Loading AA sequences from ATLAS...")
    seq_map = _load_sequences(accessions)
    n_seqs  = sum(1 for a in accessions if a in seq_map)
    print(f"  Sequences found: {n_seqs}/{len(accessions)}")

    records: list[dict] = []
    for i, acc in enumerate(accessions, 1):
        if i % 25 == 0 or i == 1:
            print(f"  {i}/{len(accessions)}  ({acc})")

        zero_filled = False
        plddt       = 0.0
        emb         = np.zeros(EMBED_DIM, dtype=np.float32)

        # 1. Download / cache AF PDB
        pdb_path = _fetch_af_pdb(acc)
        if pdb_path is None:
            zero_filled = True
        else:
            # 2. pLDDT gate
            plddt = _mean_plddt(pdb_path)
            if plddt < PLDDT_THRESH:
                zero_filled = True
            else:
                # 3. AA sequence
                seq_aa = seq_map.get(acc, "")
                if not seq_aa:
                    print(f"    [WARN] no sequence in ATLAS for {acc}; zero-filling")
                    zero_filled = True
                else:
                    # 4. Foldseek 3Di tokens
                    tokens_3di = _foldseek_3di(pdb_path)
                    if tokens_3di is None:
                        print(f"    [WARN] Foldseek failed for {acc}; zero-filling")
                        zero_filled = True
                    else:
                        # 5-6. SaProt embedding
                        try:
                            emb = _embed(seq_aa, tokens_3di, tokenizer, model, device)
                        except Exception as exc:
                            print(f"    [WARN] SaProt embed failed for {acc}: {exc}")
                            zero_filled = True
                            emb = np.zeros(EMBED_DIM, dtype=np.float32)

        rec = {ACC_COL: acc, "plddt_mean": plddt, "zero_filled": zero_filled}
        for j, v in enumerate(emb):
            rec[f"struct_{j}"] = float(v)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(OUT, compression="zstd")

    n_zero = int(df["zero_filled"].sum())
    print(f"\n=== SaProt complete ===")
    print(f"Proteins:          {len(df)}")
    print(f"Zero-filled:       {n_zero}  ({n_zero/len(df)*100:.1f}%)")
    valid = df[~df["zero_filled"]]
    if len(valid):
        print(f"Mean pLDDT (valid):{valid['plddt_mean'].mean():.1f}")
    print(f"Output -> {OUT}")

    # Collapse audit
    # Audit only non-zero-filled proteins; zero-filled rows are intentional
    # placeholders (no AF structure / pLDDT < 70) and share the zero vector by design.
    struct_cols = [f"struct_{j}" for j in range(EMBED_DIM)]
    valid_df    = df[~df["zero_filled"]]
    embs        = valid_df[struct_cols].values.astype(np.float32)
    n_valid     = len(valid_df)
    n_uniq      = len(np.unique(embs.round(4), axis=0)) if n_valid > 0 else 0
    ratio       = n_uniq / n_valid if n_valid > 0 else 0.0
    print(f"\nCollapse audit: {n_uniq}/{n_valid} unique rows among valid embeddings  ({ratio*100:.1f}%)")
    if ratio < 0.95:
        raise RuntimeError(
            f"Embedding collapse: only {ratio*100:.1f}% unique rows (threshold 95%). "
            f"Check Foldseek 3Di output and SaProt tokenizer."
        )
    print("Collapse audit PASS (>= 95% unique among valid embeddings).")


if __name__ == "__main__":
    run()
