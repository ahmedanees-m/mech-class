"""Public Predictor API for mech-class v0.5.0.

Loads trained LightGBM models (stored as plain dicts from training scripts) and
exposes a clean Python API for mechanism prediction from sequence.

Feature pipeline at inference time:
  F_seq   (640d)  — ESM-2 150M mean-pool, lazy-loaded singleton, CPU-only
  F_struct (1280d) — zero-filled (requires SaProt + GPU; optional via pdb_path)
  F_domain  (26d)  — PFAM_WHITELIST binary flags + composite/single-domain flags
                     UniProt REST lookup if accession given, else zero-filled
  F_active_site (7d) — zero-filled (requires PDB structure geometry)

Total: 1953-dim vector matching the training feature_matrix.parquet columns.

Usage:
    predictor = Predictor.load("/path/to/model/dir")
    pred = predictor.predict_from_sequence("Q99ZW2", "MDKKY...")
    print(pred.tier_a, pred.composite, pred.composite_prob)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel

# Pfam whitelist — identical to training (dom_0..dom_22)
PFAM_WHITELIST = [
    "PF13395",
    "PF18541",
    "PF16595",
    "PF18516",
    "PF01548",
    "PF02371",
    "PF07282",
    "PF00665",
    "PF01609",
    "PF13586",
    "PF08721",
    "PF11426",
    "PF05621",
    "PF00589",
    "PF00239",
    "PF07508",
    "PF01844",
    "PF02486",
    "PF18061",
    "PF16592",
    "PF16593",
    "PF13639",
    "PF03377",
]

# Zenodo deposit URL for trained model artifacts (v1.0 release)
# Set MECH_CLASS_MODEL_DIR env var to override local cache path.
_ZENODO_RECORD_URL = "https://zenodo.org/records/TODO_FILL_AFTER_ZENODO_DEPOSIT"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mech-class" / "models" / "v1.0"


class Prediction(BaseModel):
    """Mechanism prediction for a single protein."""

    accession: str
    sequence_length: int
    tier_a: str
    tier_a_confidence: float
    tier_b: str | None = None
    tier_b_confidence: float | None = None
    composite: bool = False
    composite_prob: float = 0.0
    composite_evidence: list[str] = []
    pfam_hits: list[str] = []
    channels_used: list[str] = []

    @property
    def confidence(self) -> float:
        return self.tier_a_confidence

    def summary(self) -> str:
        """One-line human-readable summary."""
        comp = f" [COMPOSITE P={self.composite_prob:.3f}]" if self.composite else ""
        tb = f" / {self.tier_b}" if self.tier_b else ""
        return f"{self.accession}: {self.tier_a}{tb} (conf={self.tier_a_confidence:.3f}){comp}"


class Predictor:
    """Load trained MECH-CLASS models and predict mechanism from sequence.

    Model artifacts are plain pickled dicts produced by training scripts:
      tier_a/model.pkl        → {"model": LGBMClassifier, "feature_cols": list,
                                  "label_encoder": LabelEncoder}
      composite_head/model.pkl→ {"model": LGBMClassifier, "feature_cols": list}
      tier_b/{CLASS}/model.pkl→ {"model": LGBMClassifier, "feature_cols": list,
                                  "label_encoder": LabelEncoder}

    Parameters
    ----------
    _ta : dict
        Tier-A model dict.
    _comp : dict
        Composite head model dict.
    _tier_b : dict[str, dict]
        Map from Tier-A class name to Tier-B model dict.
    _esm2_singleton : callable or None
        Lazy ESM-2 embed function (populated on first call).
    """

    def __init__(self, _ta: dict, _comp: dict, _tier_b: dict):
        self._ta = _ta
        self._comp = _comp
        self._tier_b = _tier_b
        self._esm2 = None  # loaded lazily

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_dir: str | Path | None = None,
        *,
        download: bool = True,
    ) -> Predictor:
        """Load trained models from a local directory.

        Parameters
        ----------
        model_dir : path-like, optional
            Directory containing ``tier_a/model.pkl``,
            ``composite_head/model.pkl``, and ``tier_b/*/model.pkl``.
            Defaults to ``~/.cache/mech-class/models/v1.0/`` (Zenodo cache).
        download : bool
            If True and model_dir is the default cache, attempt to download
            models from Zenodo if not already cached.  Set False to skip.
        """
        if model_dir is None:
            model_dir = _DEFAULT_CACHE_DIR
            if download and not (Path(model_dir) / "tier_a" / "model.pkl").exists():
                _download_from_zenodo(Path(model_dir))
        model_dir = Path(model_dir)

        ta_path = model_dir / "tier_a" / "model.pkl"
        comp_path = model_dir / "composite_head" / "model.pkl"

        if not ta_path.exists():
            raise FileNotFoundError(
                f"Tier-A model not found at {ta_path}.\n"
                "Pass model_dir= pointing to your trained model directory, e.g.:\n"
                "  Predictor.load('/path/to/pen-stack/data/models')"
            )

        with open(ta_path, "rb") as f:
            ta = pickle.load(f)
        with open(comp_path, "rb") as f:
            comp = pickle.load(f)

        # Load per-class Tier-B models
        tier_b: dict[str, dict] = {}
        for class_name in ta["label_encoder"].classes_:
            tb_path = model_dir / "tier_b" / class_name / "model.pkl"
            if tb_path.exists():
                with open(tb_path, "rb") as f:
                    tier_b[class_name] = pickle.load(f)

        return cls(ta, comp, tier_b)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_from_sequence(
        self,
        accession: str | None,
        sequence: str,
        *,
        pfam_hits: list[str] | None = None,
        pdb_path: str | Path | None = None,
    ) -> Prediction:
        """Predict mechanism from protein sequence.

        Parameters
        ----------
        accession : str or None
            UniProt accession.  If given, Pfam hits are looked up from UniProt
            REST API (unless ``pfam_hits`` is already supplied).
        sequence : str
            Amino acid sequence.
        pfam_hits : list[str], optional
            Pre-computed Pfam accessions (overrides UniProt lookup).
        pdb_path : path-like, optional
            Path to PDB/CIF file for structure-channel features (not yet
            implemented; channel zero-filled).

        Returns
        -------
        Prediction
        """
        acc_str = accession or "unknown"

        # --- Pfam domain hits -------------------------------------------
        if pfam_hits is None:
            if accession:
                pfam_hits = _fetch_pfam_hits(accession)
            else:
                pfam_hits = []

        channels_used: list[str] = []

        # --- F_seq channel (ESM-2 150M) ---------------------------------
        seq_emb = self._embed_sequence(sequence)
        if seq_emb is not None:
            channels_used.append("F_seq")
        else:
            seq_emb = np.zeros(640, dtype=np.float32)

        # --- F_struct channel (zero-filled; optional SaProt) ------------
        # pdb_path support deferred to v0.6.0; always zero-fill for now.
        # channels_used.append("F_struct")  # uncomment when implemented

        # --- Build feature DataFrame ------------------------------------
        feat_cols = self._ta["feature_cols"]
        X_df = _build_feature_row(seq_emb, pfam_hits, feat_cols)

        # --- Tier-A prediction ------------------------------------------
        proba_a = self._ta["model"].predict_proba(X_df)[0]
        pred_idx = int(np.argmax(proba_a))
        tier_a = self._ta["label_encoder"].inverse_transform([pred_idx])[0]
        tier_a_cf = float(proba_a[pred_idx])

        # --- Composite head ---------------------------------------------
        comp_feat_cols = self._comp.get("feature_cols") or feat_cols
        X_comp = X_df[comp_feat_cols] if comp_feat_cols else X_df
        comp_proba = self._comp["model"].predict_proba(X_comp)[0]
        composite = bool(comp_proba[1] >= 0.5)
        composite_prob = float(comp_proba[1])

        # Build composite evidence strings
        comp_ev: list[str] = []
        if composite:
            pfam_set = set(pfam_hits)
            if "PF01548" in pfam_set:
                comp_ev.append("RuvC-fold DEDD N-terminal domain (PF01548)")
            if "PF02371" in pfam_set:
                comp_ev.append("Serine Tnp C-terminal domain (PF02371)")
            if not comp_ev:
                comp_ev.append(f"Composite score P={composite_prob:.3f} (multi-domain heuristic)")

        # --- Tier-B prediction ------------------------------------------
        tier_b_label: str | None = None
        tier_b_cf: float | None = None
        if tier_a in self._tier_b:
            tb = self._tier_b[tier_a]
            X_tb = X_df[tb["feature_cols"]] if tb.get("feature_cols") else X_df
            pb_b = tb["model"].predict_proba(X_tb)[0]
            idx_b = int(np.argmax(pb_b))
            tier_b_label = tb["label_encoder"].inverse_transform([idx_b])[0]
            tier_b_cf = float(pb_b[idx_b])

        if pfam_hits:
            channels_used.append("F_domain")

        return Prediction(
            accession=acc_str,
            sequence_length=len(sequence),
            tier_a=tier_a,
            tier_a_confidence=tier_a_cf,
            tier_b=tier_b_label,
            tier_b_confidence=tier_b_cf,
            composite=composite,
            composite_prob=composite_prob,
            composite_evidence=comp_ev,
            pfam_hits=pfam_hits,
            channels_used=channels_used,
        )

    def predict_from_fasta(self, fasta_path: str | Path) -> list[Prediction]:
        """Predict mechanism for all sequences in a FASTA file."""
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("biopython required: pip install biopython")
        results = []
        for rec in SeqIO.parse(str(fasta_path), "fasta"):
            pred = self.predict_from_sequence(rec.id, str(rec.seq))
            results.append(pred)
        return results

    def predict_batch(
        self,
        df: pd.DataFrame,
        *,
        pfam_col: str | None = "pfam_hits",
    ) -> pd.DataFrame:
        """Predict for a DataFrame with columns: accession, sequence [, pfam_hits].

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'accession' and 'sequence' columns.
        pfam_col : str or None
            Column name containing pre-computed Pfam hit lists.
            Pass None to force UniProt lookup for every row.
        """
        results = []
        for _, row in df.iterrows():
            pfam = row[pfam_col] if (pfam_col and pfam_col in row.index) else None
            if isinstance(pfam, float):  # NaN → None
                pfam = None
            p = self.predict_from_sequence(
                row.get("accession"),
                row["sequence"],
                pfam_hits=pfam,
            )
            results.append(p.model_dump())
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_sequence(self, sequence: str) -> np.ndarray | None:
        """Return ESM-2 150M mean-pool embedding, or None on failure."""
        if self._esm2 is None:
            self._esm2 = _load_esm2_singleton()
        if self._esm2 is None:
            return None
        model, alphabet, batch_converter = self._esm2
        try:
            import torch

            seq = sequence[:1022]
            _, _, tokens = batch_converter([("q", seq)])
            with torch.no_grad():
                out = model(tokens, repr_layers=[30])
            emb = out["representations"][30][0, 1 : len(seq) + 1].mean(0)
            return emb.cpu().numpy().astype(np.float32)
        except Exception as exc:
            import warnings

            warnings.warn(f"ESM-2 embedding failed ({exc}); F_seq zero-filled.")
            return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_ESM2_SINGLETON = None


def _load_esm2_singleton():
    """Lazy-load ESM-2 150M once per process."""
    global _ESM2_SINGLETON
    if _ESM2_SINGLETON is not None:
        return _ESM2_SINGLETON
    try:
        import esm as fair_esm

        model, alphabet = fair_esm.pretrained.esm2_t30_150M_UR50D()
        model = model.eval()
        batch_converter = alphabet.get_batch_converter()
        _ESM2_SINGLETON = (model, alphabet, batch_converter)
        return _ESM2_SINGLETON
    except Exception:
        return None


def _fetch_pfam_hits(accession: str, timeout: int = 15) -> list[str]:
    """Query UniProt REST API for Pfam cross-references of a protein.

    Falls back to empty list on any network or parsing error.
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        pfam = []
        for ref in data.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "Pfam":
                pfam.append(ref["id"])
        return pfam
    except Exception:
        return []


def _build_feature_row(
    seq_emb: np.ndarray,
    pfam_hits: list[str],
    feat_cols: list[str],
) -> pd.DataFrame:
    """Assemble a 1-row feature DataFrame matching training feature_cols.

    Maps:
      seq_0..639      ← ESM-2 embedding
      struct_0..1279  ← zero-filled (SaProt not available at inference time)
      dom_0..22       ← PFAM_WHITELIST binary flags
      dom_23          ← IS110 composite (PF01548 AND PF02371)
      dom_24          ← editor fusion flag (reserved, zero)
      dom_25          ← single-domain flag
      as_0..6         ← zero-filled (active-site geometry)
    """
    row = np.zeros(len(feat_cols), dtype=np.float32)
    col_map = {c: i for i, c in enumerate(feat_cols)}

    # F_seq channel
    for k, v in enumerate(seq_emb):
        c = f"seq_{k}"
        if c in col_map:
            row[col_map[c]] = float(v)

    # F_domain channel
    pfam_set = set(pfam_hits)
    wl_hits: list[str] = []
    for wl_idx, pfam in enumerate(PFAM_WHITELIST):
        c = f"dom_{wl_idx}"
        if c in col_map and pfam in pfam_set:
            row[col_map[c]] = 1.0
            wl_hits.append(pfam)

    # IS110 composite flag (dom_23)
    if "dom_23" in col_map:
        row[col_map["dom_23"]] = float("PF01548" in pfam_set and "PF02371" in pfam_set)
    # dom_24 = editor fusion (zero; reserved for future)
    # Single-domain flag (dom_25)
    if "dom_25" in col_map:
        row[col_map["dom_25"]] = float(len(wl_hits) == 1)

    return pd.DataFrame(row.reshape(1, -1), columns=feat_cols)


def _download_from_zenodo(target_dir: Path) -> None:
    """Attempt to download model artifacts from Zenodo.

    This is a stub for v0.5.0 — the Zenodo deposit URL will be filled in
    after uploading the trained models following peer review.
    """
    raise RuntimeError(
        "Zenodo model download not yet configured for mech-class v0.5.0.\n"
        "Please pass model_dir= explicitly:\n\n"
        "  predictor = Predictor.load('/path/to/pen-stack/data/models')\n\n"
        "Trained model artifacts will be deposited to Zenodo upon publication."
    )
