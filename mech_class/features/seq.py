"""F_seq channel: ESM-2 150M sequence embeddings.

Two usage modes:
  1. Training / batch  — reuse Paper 1's esm2_150M_v6.parquet (640-dim vectors).
  2. Inference         — run ESM-2 150M on-the-fly via lazy-loaded singleton.

The model is loaded once per process (singleton pattern) and kept in memory.
On CPU, inference takes ~5–30 s per sequence depending on length.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ESM2_DIM        = 640
ESM2_PARQUET    = Path("/data/embeddings/esm2_150M_v6.parquet")
ESM2_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM2_REPR_LAYER = 30
ESM2_MAX_LEN    = 1022   # ESM-2 hard token limit (excludes BOS/EOS)

# Process-level singleton
_ESM2_MODEL      = None
_ESM2_ALPHABET   = None
_ESM2_CONVERTER  = None


def load_esm2_singleton(*, verbose: bool = True) -> bool:
    """Load ESM-2 150M into the module singleton.  Returns True on success."""
    global _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_CONVERTER
    if _ESM2_MODEL is not None:
        return True
    try:
        import esm as fair_esm
        model, alphabet = fair_esm.pretrained.esm2_t30_150M_UR50D()
        model = model.eval()
        _ESM2_MODEL     = model
        _ESM2_ALPHABET  = alphabet
        _ESM2_CONVERTER = alphabet.get_batch_converter()
        if verbose:
            print(f"[mech-class] ESM-2 150M loaded (CPU mode, repr_layer={ESM2_REPR_LAYER})")
        return True
    except ImportError:
        warnings.warn(
            "fair-esm not installed. F_seq channel will be zero-filled.\n"
            "Install with: pip install fair-esm"
        )
        return False
    except Exception as exc:
        warnings.warn(f"ESM-2 load failed: {exc}. F_seq channel will be zero-filled.")
        return False


def embed_sequence(sequence: str, *, device: str = "cpu") -> np.ndarray:
    """Embed a single sequence with ESM-2 150M (mean-pool, repr_layer=30).

    Lazy-loads the model singleton on first call.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.  Truncated to ESM2_MAX_LEN (1022 aa).
    device : str
        'cpu' or 'cuda'.  GPU is faster (~0.5 s vs ~10 s for 500-aa protein).

    Returns
    -------
    np.ndarray, shape (640,), dtype float32
        Mean-pooled token representations.
        Returns zero vector if ESM-2 is unavailable.
    """
    if _ESM2_MODEL is None:
        load_esm2_singleton(verbose=False)

    if _ESM2_MODEL is None:
        return np.zeros(ESM2_DIM, dtype=np.float32)

    try:
        import torch
        seq = sequence[:ESM2_MAX_LEN]
        model = _ESM2_MODEL
        if device != "cpu":
            model = model.to(device)

        _, _, tokens = _ESM2_CONVERTER([("q", seq)])
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[ESM2_REPR_LAYER])

        emb = out["representations"][ESM2_REPR_LAYER][0, 1 : len(seq) + 1].mean(0)
        return emb.cpu().numpy().astype(np.float32)

    except Exception as exc:
        warnings.warn(f"ESM-2 inference failed ({exc}); returning zero vector.")
        return np.zeros(ESM2_DIM, dtype=np.float32)


# Legacy alias for backward compatibility
extract_esm2_features = embed_sequence


def load_esm2_embeddings(path: Path = ESM2_PARQUET) -> pd.DataFrame:
    """Load Paper 1 precomputed ESM-2 embeddings parquet.

    Returns DataFrame with columns: accession, embedding (list of 640 floats).
    """
    df = pd.read_parquet(path)
    if "accession" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"ESM-2 parquet at {path} is missing 'accession' or 'embedding' column")
    return df


def get_esm2_vector(accession: str, df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Look up a precomputed ESM-2 embedding by accession.

    Parameters
    ----------
    accession : str
        UniProt accession.
    df : pd.DataFrame, optional
        Preloaded embeddings dataframe.  Loaded from ESM2_PARQUET if None.

    Returns
    -------
    np.ndarray, shape (640,), dtype float32

    Raises
    ------
    KeyError
        If the accession is not found in the embeddings file.
    """
    if df is None:
        df = load_esm2_embeddings()
    row = df[df["accession"] == accession]
    if row.empty:
        raise KeyError(f"Accession {accession!r} not in ESM-2 embeddings parquet.")
    return np.asarray(row.iloc[0]["embedding"], dtype=np.float32)


def build_seq_feature_matrix(
    accessions: list[str],
    *,
    esm2_df: Optional[pd.DataFrame] = None,
    allow_inference: bool = False,
    sequences: Optional[dict[str, str]] = None,
) -> np.ndarray:
    """Build N × 640 ESM-2 feature matrix from accession list.

    Parameters
    ----------
    accessions : list[str]
        UniProt accessions.
    esm2_df : pd.DataFrame, optional
        Preloaded embeddings (avoids disk re-read on repeated calls).
    allow_inference : bool
        If True and a sequence is missing from the parquet, run inference
        using the singleton model (requires ``sequences`` dict).
    sequences : dict[str, str], optional
        Map accession → amino-acid sequence for inference fallback.

    Returns
    -------
    np.ndarray, shape (N, 640), dtype float32
    """
    if esm2_df is None:
        esm2_df = load_esm2_embeddings()

    matrix  = np.zeros((len(accessions), ESM2_DIM), dtype=np.float32)
    missing: list[str] = []

    acc_to_emb = dict(
        zip(esm2_df["accession"].tolist(),
            esm2_df["embedding"].tolist())
    )

    for i, acc in enumerate(accessions):
        if acc in acc_to_emb:
            matrix[i] = np.asarray(acc_to_emb[acc], dtype=np.float32)
        else:
            missing.append(acc)
            if allow_inference and sequences and acc in sequences:
                matrix[i] = embed_sequence(sequences[acc])

    if missing:
        warnings.warn(
            f"{len(missing)} accessions missing ESM-2 embeddings "
            f"(zero-filled): {missing[:5]}"
        )

    return matrix
