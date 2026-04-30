"""Prospective prediction: Fanzor/OMEGA-nuclease catalog.

Runs mech-class predictor over candidate Fanzor ortholog sequences from the
genome-atlas (Eukaryota proteins + TnpB/PF07282-domain proteins).

Outputs a ranked catalog with:
  - tier_a, tier_b predictions + confidences
  - IS110 composite flag
  - Source evidence (which atlas search hit nominated this sequence)
  - Novelty score (cosine distance from nearest training example in ESM-2 space)

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/40_predict_fanzor_catalog.py"

Inputs:
  /data/graphs/atlas.duckdb                   -- protein metadata + domain edges
  /data/features/fused/feature_matrix.parquet -- training features (for novelty scoring)

Expected output:
  /data/results/fanzor_catalog.parquet
  /data/results/fanzor_catalog.tsv
  /data/results/fanzor_catalog_summary.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths ? all inside /data mount
# ---------------------------------------------------------------------------
ATLAS_DB       = Path("/data/graphs/atlas.duckdb")
FEAT_MATRIX    = Path("/data/features/fused/feature_matrix.parquet")
MODEL_DIR      = Path("/data/models")
TIER_A_PATH    = MODEL_DIR / "tier_a" / "model.pkl"
COMP_PATH      = MODEL_DIR / "composite_head" / "model.pkl"
TIER_B_DIR     = MODEL_DIR / "tier_b"
CHECKPOINT_DIR = Path("/data/results/fanzor_catalog_ckpt")
OUT_DIR        = Path("/data/results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FANZOR_CONF_THRESHOLD = 0.70
CHECKPOINT_EVERY      = 200
PRINT_EVERY           = 50

# Pfam whitelist ? same order as dom_0..dom_22 in feature matrix
PFAM_WHITELIST = [
    "PF13395", "PF18541", "PF16595", "PF18516", "PF01548", "PF02371",
    "PF07282", "PF00665", "PF01609", "PF13586", "PF08721", "PF11426",
    "PF05621", "PF00589", "PF00239", "PF07508", "PF01844", "PF02486",
    "PF18061", "PF16592", "PF16593", "PF13639", "PF03377",
]

# Training accessions to exclude from catalog (avoid self-prediction)
TRAINING_ACCS: set[str] = set()

# ---------------------------------------------------------------------------
# ESM-2 singleton
# ---------------------------------------------------------------------------
_ESM2_MODEL = None
_ESM2_ALPHABET = None
_ESM2_BATCH_CONVERTER = None


def _load_esm2() -> None:
    global _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_BATCH_CONVERTER
    if _ESM2_MODEL is not None:
        return
    try:
        import esm as fair_esm  # fair-esm
        _ESM2_MODEL, _ESM2_ALPHABET = fair_esm.pretrained.esm2_t30_150M_UR50D()
        _ESM2_MODEL = _ESM2_MODEL.eval()
        _ESM2_BATCH_CONVERTER = _ESM2_ALPHABET.get_batch_converter()
        _log("ESM-2 150M loaded.")
    except Exception as exc:
        _log(f"[WARN] ESM-2 load failed: {exc}. Sequence embeddings will be zero-filled.")


def _embed_sequence(seq: str) -> np.ndarray:
    """Return mean-pool ESM-2 embedding (640-dim). Returns zeros on failure."""
    if _ESM2_MODEL is None:
        return np.zeros(640, dtype=np.float32)
    import torch
    try:
        seq = seq[:1022]  # ESM-2 max token length
        batch = [("x", seq)]
        _, _, tokens = _ESM2_BATCH_CONVERTER(batch)
        with torch.no_grad():
            out = _ESM2_MODEL(tokens, repr_layers=[30])
        emb = out["representations"][30][0, 1:-1].mean(0).cpu().numpy().astype(np.float32)
        return emb
    except Exception as exc:
        _log(f"[WARN] ESM-2 embed failed ({exc}). Using zeros.")
        return np.zeros(640, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _novelty_score(emb: np.ndarray, train_embs: np.ndarray) -> float:
    """Cosine distance to nearest training neighbor (1.0 = maximally novel)."""
    if train_embs.shape[0] == 0:
        return 1.0
    q = emb / (np.linalg.norm(emb) + 1e-8)
    norms = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    return float(1.0 - (norms @ q).max())


def _load_models():
    """Load Tier-A, composite head, and available Tier-B models."""
    with open(TIER_A_PATH, "rb") as f:
        ta = pickle.load(f)
    lgbm_a   = ta["model"]
    feat_cols = ta["feature_cols"]
    le_a      = ta["label_encoder"]

    with open(COMP_PATH, "rb") as f:
        comp = pickle.load(f)
    lgbm_comp   = comp["model"]
    comp_feat_cols = comp["feature_cols"]

    tier_b_models: dict[str, tuple] = {}
    if TIER_B_DIR.exists():
        for pkl in TIER_B_DIR.glob("*/model.pkl"):
            class_name = pkl.parent.name
            with open(pkl, "rb") as f:
                tb = pickle.load(f)
            tier_b_models[class_name] = (tb["model"], tb["feature_cols"], tb["label_encoder"])

    _log(f"Tier-A feature cols: {len(feat_cols)}")
    _log(f"Tier-B classes loaded: {list(tier_b_models.keys())}")
    return lgbm_a, feat_cols, le_a, lgbm_comp, comp_feat_cols, tier_b_models


def _load_candidates() -> pd.DataFrame:
    """
    Query the atlas for Fanzor candidate proteins:
      1. All Eukaryota proteins (Fanzor is eukaryotic)
      2. All proteins with PF07282 (TnpB / OMEGA-nuclease domain) regardless of kingdom

    Returns deduplicated DataFrame with columns:
      accession, sequence, organism, source_nomination
    """
    _log("Querying atlas for candidates...")
    con = duckdb.connect(str(ATLAS_DB), read_only=True)

    # Eukaryota proteins (lineage stores NCBI taxon IDs; Eukaryota = 2759)
    euk_df = con.execute("""
        SELECT
            p.accession,
            p.sequence,
            o.scientific_name AS organism,
            'eukaryota_atlas' AS source_nomination
        FROM nodes_protein p
        JOIN nodes_organism o ON p.organism_id = o.id
        WHERE list_contains(o.lineage, '2759')
          AND p.sequence IS NOT NULL AND p.sequence != ''
    """).df()
    _log(f"  Eukaryota proteins: {len(euk_df)}")

    # TnpB/PF07282-domain proteins (any kingdom)
    # edges: source_type='Protein' (capitalized), source_id = nodes_protein.id (integer)
    #        target_type='Domain' (capitalized), target_id = nodes_domain.id (integer)
    tnpb_df = con.execute("""
        SELECT
            p.accession,
            p.sequence,
            o.scientific_name AS organism,
            'tnpb_domain' AS source_nomination
        FROM nodes_protein p
        JOIN nodes_organism o ON p.organism_id = o.id
        JOIN edges e
            ON e.source_id = p.id AND e.source_type = 'Protein'
        JOIN nodes_domain d
            ON d.id = e.target_id AND e.target_type = 'Domain'
        WHERE d.accession = 'PF07282'
          AND p.sequence IS NOT NULL AND p.sequence != ''
    """).df()
    _log(f"  TnpB/PF07282 proteins: {len(tnpb_df)}")

    con.close()

    combined = pd.concat([euk_df, tnpb_df], ignore_index=True)
    # Keep first occurrence (prefer eukaryota label over tnpb when both)
    combined = combined.drop_duplicates(subset=["accession"], keep="first")
    # Exclude training proteins
    combined = combined[~combined["accession"].isin(TRAINING_ACCS)].reset_index(drop=True)
    _log(f"  Total candidates after dedup + training exclusion: {len(combined)}")
    return combined


def _get_atlas_domains(accession: str, con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return list of Pfam accessions for this protein from the atlas edges.

    edges.source_id is the integer nodes_protein.id, not the accession string.
    edges.source_type = 'Protein' (capitalized), target_type = 'Domain'.
    """
    rows = con.execute("""
        SELECT d.accession AS pfam_acc
        FROM nodes_protein p
        JOIN edges e ON e.source_id = p.id AND e.source_type = 'Protein'
        JOIN nodes_domain d ON d.id = e.target_id AND e.target_type = 'Domain'
        WHERE p.accession = ?
    """, [accession]).fetchall()
    return [r[0] for r in rows]


def _build_feature_row(
    seq_emb: np.ndarray,
    pfam_hits: list[str],
    feat_cols: list[str],
) -> np.ndarray:
    """Build a 1-row feature array matching feat_cols order.

    Channels used here:
      seq_0..639  ? ESM-2 mean-pool (640-dim)
      struct_0..1279 ? zero-filled (no AlphaFold available in batch mode)
      dom_0..22   ? Pfam whitelist binary flags
      dom_23      ? IS110 composite (PF01548 AND PF02371)
      dom_24      ? zero (editor fusion; not applicable here)
      dom_25      ? single-domain flag (exactly 1 whitelist hit)
      as_0..N     ? zero-filled (no active-site geometry available)
    """
    row = np.zeros(len(feat_cols), dtype=np.float32)

    col_map = {c: i for i, c in enumerate(feat_cols)}

    # Sequence embedding
    for k in range(640):
        col = f"seq_{k}"
        if col in col_map:
            row[col_map[col]] = seq_emb[k]

    # Domain flags
    pfam_set = set(pfam_hits)
    wl_hits = []
    for wl_idx, pfam in enumerate(PFAM_WHITELIST):
        col = f"dom_{wl_idx}"
        if col in col_map and pfam in pfam_set:
            row[col_map[col]] = 1.0
            wl_hits.append(pfam)

    # dom_23: IS110 composite
    if "dom_23" in col_map:
        row[col_map["dom_23"]] = float("PF01548" in pfam_set and "PF02371" in pfam_set)

    # dom_24: editor fusion ? leave 0

    # dom_25: single-domain
    if "dom_25" in col_map:
        row[col_map["dom_25"]] = float(len(wl_hits) == 1)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    # ---- Load training accessions for exclusion ----
    global TRAINING_ACCS
    if FEAT_MATRIX.exists():
        fm = pd.read_parquet(FEAT_MATRIX, columns=["uniprot_acc"])
        TRAINING_ACCS = set(fm["uniprot_acc"].tolist())
        _log(f"Training proteins loaded for exclusion: {len(TRAINING_ACCS)}")

    # ---- Load training ESM-2 embeddings for novelty scoring ----
    train_embs = np.zeros((0, 640), dtype=np.float32)
    if FEAT_MATRIX.exists():
        fm_full = pd.read_parquet(FEAT_MATRIX)
        seq_cols = [c for c in fm_full.columns if c.startswith("seq_")]
        if seq_cols:
            train_embs = fm_full[seq_cols].values.astype(np.float32)
            _log(f"Training ESM-2 matrix: {train_embs.shape}")

    # ---- Load models ----
    lgbm_a, feat_cols, le_a, lgbm_comp, comp_feat_cols, tier_b_models = _load_models()

    # ---- Load ESM-2 ----
    _load_esm2()

    # ---- Load candidates ----
    candidates = _load_candidates()
    n_total = len(candidates)
    _log(f"Candidate sequences to process: {n_total}")

    # ---- Check for checkpoint ----
    ckpt_path = CHECKPOINT_DIR / "progress.parquet"
    done_accs: set[str] = set()
    records: list[dict] = []
    if ckpt_path.exists():
        ckpt_df = pd.read_parquet(ckpt_path)
        records = ckpt_df.to_dict("records")
        done_accs = set(ckpt_df["accession"].tolist())
        _log(f"Resuming from checkpoint: {len(done_accs)} already done.")

    # ---- Open atlas connection (kept open for domain queries) ----
    atlas_con = duckdb.connect(str(ATLAS_DB), read_only=True)

    fanzor_count = 0

    for i, row in candidates.iterrows():
        acc = row["accession"]
        if acc in done_accs:
            continue

        seq = str(row.get("sequence", "") or "")

        # --- ESM-2 embedding ---
        seq_emb = _embed_sequence(seq) if seq else np.zeros(640, dtype=np.float32)

        # --- Domain features from atlas ---
        pfam_hits = _get_atlas_domains(acc, atlas_con)

        # --- Build feature row ---
        x = _build_feature_row(seq_emb, pfam_hits, feat_cols).reshape(1, -1)
        x_df = pd.DataFrame(x, columns=feat_cols)

        # --- Tier-A prediction ---
        proba_a = lgbm_a.predict_proba(x_df)[0]
        pred_idx = int(np.argmax(proba_a))
        tier_a = le_a.inverse_transform([pred_idx])[0]
        tier_a_conf = float(proba_a[pred_idx])

        # --- Composite head ---
        x_comp = x_df[comp_feat_cols] if comp_feat_cols else x_df
        comp_proba = lgbm_comp.predict_proba(x_comp)[0]
        composite = bool(comp_proba[1] >= 0.5)

        # --- Tier-B prediction ---
        tier_b = None
        if tier_a in tier_b_models:
            lgbm_b, b_feat_cols, le_b = tier_b_models[tier_a]
            x_b = x_df[b_feat_cols] if b_feat_cols else x_df
            proba_b = lgbm_b.predict_proba(x_b)[0]
            b_idx = int(np.argmax(proba_b))
            tier_b = le_b.inverse_transform([b_idx])[0]

        # --- Novelty score ---
        novelty = _novelty_score(seq_emb, train_embs) if train_embs.shape[0] > 0 else 1.0

        rec = {
            "accession":         acc,
            "tier_a":            tier_a,
            "tier_a_confidence": tier_a_conf,
            "tier_b":            tier_b,
            "composite":         composite,
            "novelty_score":     novelty,
            "organism":          str(row.get("organism", "") or ""),
            "source_nomination": str(row.get("source_nomination", "atlas_query") or "atlas_query"),
        }
        records.append(rec)
        done_accs.add(acc)

        if tier_a == "DSB_NUCLEASE" and tier_b and "Fanzor" in tier_b and tier_a_conf >= FANZOR_CONF_THRESHOLD:
            fanzor_count += 1

        processed = len(done_accs)
        if processed % PRINT_EVERY == 0:
            _log(f"  {processed}/{n_total} processed  (Fanzor so far: {fanzor_count})")

        if processed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_parquet(ckpt_path, compression="zstd")
            _log(f"  [checkpoint] {processed} records saved.")

    atlas_con.close()

    # ---- Assemble final output ----
    if not records:
        _log("[WARN] No records produced. Check atlas candidate query.")
        return

    df = pd.DataFrame(records)

    # Sort: Fanzor high-conf first, then others by tier-A confidence
    fanzor_mask = (df["tier_a"] == "DSB_NUCLEASE") & df["tier_b"].str.contains("Fanzor", na=False)
    df_fanzor = df[fanzor_mask].sort_values("tier_a_confidence", ascending=False)
    df_other  = df[~fanzor_mask].sort_values("tier_a_confidence", ascending=False)
    df_out    = pd.concat([df_fanzor, df_other], ignore_index=True)

    df_out.to_parquet(OUT_DIR / "fanzor_catalog.parquet", compression="zstd")
    df_out.to_csv(OUT_DIR / "fanzor_catalog.tsv", sep="\t", index=False)

    # Recount with final df
    fanzor_count_final = int(
        ((df_out["tier_a"] == "DSB_NUCLEASE")
         & df_out["tier_b"].str.contains("Fanzor", na=False)
         & (df_out["tier_a_confidence"] >= FANZOR_CONF_THRESHOLD)).sum()
    )

    summary = {
        "total_candidates":       len(df_out),
        "fanzor_high_conf":       fanzor_count_final,
        "fanzor_threshold":       FANZOR_CONF_THRESHOLD,
        "tier_a_distribution":    df_out["tier_a"].value_counts().to_dict(),
        "tier_b_distribution":    df_out["tier_b"].value_counts().dropna().to_dict(),
        "composite_count":        int(df_out["composite"].sum()),
    }
    (OUT_DIR / "fanzor_catalog_summary.json").write_text(json.dumps(summary, indent=2))

    _log("")
    _log("=== Fanzor catalog complete ===")
    _log(f"Total candidates processed : {len(df_out)}")
    _log(f"Fanzor (conf >= {FANZOR_CONF_THRESHOLD})    : {fanzor_count_final}")
    _log(f"Catalog -> {OUT_DIR / 'fanzor_catalog.parquet'}")
    _log(f"TSV     -> {OUT_DIR / 'fanzor_catalog.tsv'}")


if __name__ == "__main__":
    run()
