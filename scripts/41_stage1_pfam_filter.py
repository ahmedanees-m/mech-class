"""Step 23a — Stage 1: Assemble IS110 composite candidates from UniProt.

Queries UniProt for ALL proteins annotated with BOTH:
  PF01548 (DEDD_Tnp_IS110 -- RuvC-like N-terminal of IS110-family)
  PF02371 (Transposase_20  -- serine recombinase C-terminal of IS110-family)

This dual-domain co-occurrence is the definitive IS110 composite signature.
InterPro CL0219 (RNase H-like clan) incorrectly classifies such proteins as
DSB_NUCLEASE based on the N-terminal fold alone; this triage corrects that.

UniProt returns ~31,883 entries with both domains. After training exclusion and
domain-only MECH-CLASS scoring, the top candidates by composite_prob form the
published IS110 reclassification catalog.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install lightgbm --quiet && python scripts/41_stage1_pfam_filter.py"

Expected output:
  /data/predictions/is110_triage/stage1_candidates.parquet
  /data/predictions/is110_triage/stage1_summary.json
  /data/predictions/is110_triage/triage_results.parquet
  /data/predictions/is110_triage/triage_results.tsv
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT_DIR = Path("/data/predictions/is110_triage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_FEAT = Path("/data/features/fused/feature_matrix.parquet")
MODEL_DIR     = Path("/data/models")
TIER_A_PATH   = MODEL_DIR / "tier_a" / "model.pkl"
COMP_PATH     = MODEL_DIR / "composite_head" / "model.pkl"

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Pfam whitelist (dom_0..dom_22)
PFAM_WHITELIST = [
    "PF13395", "PF18541", "PF16595", "PF18516", "PF01548", "PF02371",
    "PF07282", "PF00665", "PF01609", "PF13586", "PF08721", "PF11426",
    "PF05621", "PF00589", "PF00239", "PF07508", "PF01844", "PF02486",
    "PF18061", "PF16592", "PF16593", "PF13639", "PF03377",
]

PAGE_SIZE = 500  # UniProt pagination


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _fetch_paged(query: str, fields: str) -> pd.DataFrame:
    """Fetch all UniProt results for a query, handling pagination."""
    # First, get total count
    r_count = requests.get(
        UNIPROT_SEARCH,
        params={"query": query, "format": "tsv", "fields": "accession", "size": 1},
        timeout=30,
    )
    total = int(r_count.headers.get("X-Total-Results", 0))
    _log(f"  Total matching: {total}")

    if total == 0:
        return pd.DataFrame()

    all_dfs = []
    cursor = None
    fetched = 0
    page_num = 0

    while True:
        params = {"format": "tsv", "query": query, "fields": fields, "size": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor

        for attempt in range(3):
            try:
                r = requests.get(UNIPROT_SEARCH, params=params, timeout=120)
                r.raise_for_status()
                break
            except Exception as exc:
                _log(f"  Page {page_num} attempt {attempt+1} failed: {exc}")
                if attempt == 2:
                    raise
                time.sleep(5)

        content = r.content.decode("utf-8", errors="replace")
        df_page = pd.read_csv(io.StringIO(content), sep="\t")
        if df_page.empty:
            break

        all_dfs.append(df_page)
        fetched += len(df_page)
        page_num += 1

        if page_num % 10 == 0:
            _log(f"  Fetched {fetched}/{total} ({100*fetched//max(total,1)}%)...")

        # Pagination: next cursor from Link header
        link_header = r.headers.get("Link", "")
        if 'rel="next"' in link_header:
            import re
            m = re.search(r'cursor=([^&>]+)', link_header)
            cursor = m.group(1) if m else None
            if not cursor:
                break
        else:
            break

        if fetched >= total:
            break

    _log(f"  Fetched total: {fetched}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _build_feature_row(pfam_hits: list[str], feat_cols: list[str]) -> np.ndarray:
    """Domain-only features (IS110 reclassification is domain-driven)."""
    row = np.zeros(len(feat_cols), dtype=np.float32)
    col_map = {c: i for i, c in enumerate(feat_cols)}
    pfam_set = set(pfam_hits)
    wl_hits = []
    for wl_idx, pfam in enumerate(PFAM_WHITELIST):
        c = f"dom_{wl_idx}"
        if c in col_map and pfam in pfam_set:
            row[col_map[c]] = 1.0
            wl_hits.append(pfam)
    if "dom_23" in col_map:
        row[col_map["dom_23"]] = float("PF01548" in pfam_set and "PF02371" in pfam_set)
    if "dom_25" in col_map:
        row[col_map["dom_25"]] = float(len(wl_hits) == 1)
    return row


def run() -> None:
    # ---- Load training accessions ----
    training_accs: set[str] = set()
    if TRAINING_FEAT.exists():
        fm = pd.read_parquet(TRAINING_FEAT, columns=["uniprot_acc"])
        training_accs = set(fm["uniprot_acc"].tolist())
        _log(f"Training proteins (excluded): {len(training_accs)}")

    # ---- Stage 1: Fetch IS110 composite proteins from UniProt ----
    stage1_path = OUT_DIR / "stage1_candidates.parquet"
    if stage1_path.exists():
        _log("Loading cached Stage 1 candidates...")
        candidates = pd.read_parquet(stage1_path)
    else:
        _log("Stage 1: querying UniProt for PF01548+PF02371 proteins...")
        query = "(xref:pfam-PF01548) AND (xref:pfam-PF02371)"
        fields = "accession,protein_name,organism_name,length,xref_pfam"
        candidates = _fetch_paged(query, fields)

        if candidates.empty:
            _log("[ERROR] No Stage 1 candidates retrieved.")
            return

        # Standardize columns
        col_map = {}
        for c in candidates.columns:
            cl = c.lower()
            if cl == "entry":
                col_map[c] = "accession"
            elif "protein name" in cl:
                col_map[c] = "protein_name"
            elif "organism" in cl and "id" not in cl:
                col_map[c] = "organism"
            elif cl == "length":
                col_map[c] = "length"
            elif "pfam" in cl:
                col_map[c] = "pfam_refs"
        candidates = candidates.rename(columns=col_map)
        candidates.to_parquet(stage1_path, compression="zstd")
        _log(f"Stage 1 cached: {len(candidates)} proteins")

    _log(f"Stage 1 proteins: {len(candidates)}")

    # ---- Exclude training ----
    if "accession" in candidates.columns:
        before = len(candidates)
        candidates = candidates[~candidates["accession"].isin(training_accs)].reset_index(drop=True)
        _log(f"After training exclusion: {len(candidates)} (removed {before - len(candidates)})")

    # ---- Load models ----
    with open(TIER_A_PATH, "rb") as f:
        ta = pickle.load(f)
    lgbm_a, feat_cols, le_a = ta["model"], ta["feature_cols"], ta["label_encoder"]

    with open(COMP_PATH, "rb") as f:
        comp = pickle.load(f)
    lgbm_comp, comp_feat_cols = comp["model"], comp["feature_cols"]
    _log(f"Models loaded. Tier-A feature cols: {len(feat_cols)}")

    # ---- Build feature matrix (domain-only, vectorized) ----
    _log(f"Building domain-only feature matrix for {len(candidates)} candidates...")

    # All proteins have PF01548+PF02371 by construction; add any others from pfam_refs
    def _parse_pfam(row) -> list[str]:
        base = ["PF01548", "PF02371"]
        refs = str(row.get("pfam_refs", "") or "")
        extras = [p.strip() for p in refs.replace(";", " ").split() if p.strip().startswith("PF")]
        return list(set(base + extras))

    X = np.vstack([
        _build_feature_row(_parse_pfam(row), feat_cols)
        for _, row in candidates.iterrows()
    ])
    X_df = pd.DataFrame(X, columns=feat_cols)
    _log("Feature matrix built. Running predictions...")

    # ---- Tier-A ----
    proba_a = lgbm_a.predict_proba(X_df)
    pred_idx = np.argmax(proba_a, axis=1)
    tier_a_labels = le_a.inverse_transform(pred_idx)
    tier_a_conf = proba_a[np.arange(len(proba_a)), pred_idx]

    # ---- Composite head ----
    X_comp = X_df[comp_feat_cols] if comp_feat_cols else X_df
    comp_proba = lgbm_comp.predict_proba(X_comp)
    composite_bool = comp_proba[:, 1] >= 0.5
    composite_prob = comp_proba[:, 1]

    # ---- Assemble results ----
    # NOTE: domain-only triage -- Tier-A predicts DSB_NUCLEASE for ALL PF01548+PF02371
    # proteins with zero seq/struct (InterPro CL0219 misclassification = exactly what we correct).
    # The composite head IS the reclassification signal; Tier-A is reported for documentation.
    results = []
    n_reclassified = 0
    for i, (_, row) in enumerate(candidates.iterrows()):
        is110 = bool(composite_bool[i])   # composite head alone for domain-only triage
        if is110:
            n_reclassified += 1
        results.append({
            "accession":          row.get("accession", ""),
            "protein_name":       row.get("protein_name", ""),
            "organism":           row.get("organism", ""),
            "length":             row.get("length", None),
            "tier_a":             tier_a_labels[i],
            "tier_a_confidence":  float(tier_a_conf[i]),
            "composite":          bool(composite_bool[i]),
            "composite_prob":     float(composite_prob[i]),
            "is110_reclassified": is110,
        })

    df = pd.DataFrame(results)

    # Sort: IS110 reclassified by composite_prob desc, then others
    df_is110 = df[df["is110_reclassified"]].sort_values("composite_prob", ascending=False)
    df_other  = df[~df["is110_reclassified"]].sort_values("composite_prob", ascending=False)
    df_out    = pd.concat([df_is110, df_other], ignore_index=True)

    df_out.to_parquet(OUT_DIR / "triage_results.parquet", compression="zstd")
    df_out.to_csv(OUT_DIR / "triage_results.tsv", sep="\t", index=False)

    summary = {
        "stage1_total":              len(candidates),
        "is110_reclassified_count":  n_reclassified,
        "is110_reclassified_frac":   float(n_reclassified / max(len(candidates), 1)),
        "composite_flagged":         int(composite_bool.sum()),
        "tier_a_distribution":       df_out["tier_a"].value_counts().to_dict(),
        "feature_channels":          "domain_only (seq/struct zero-filled -- high-throughput triage)",
        "uniprot_query":             "(xref:pfam-PF01548) AND (xref:pfam-PF02371)",
    }
    (OUT_DIR / "triage_summary.json").write_text(json.dumps(summary, indent=2))

    _log("")
    _log("=== IS110 composite triage complete ===")
    _log(f"Stage 1 proteins (PF01548+PF02371): {len(candidates)}")
    _log(f"IS110 reclassified (composite=True, domain-only triage): {n_reclassified}  ({100*n_reclassified/max(len(candidates),1):.1f}%)")
    _log(f"Tier-A distribution:")
    for cls, cnt in df_out["tier_a"].value_counts().items():
        _log(f"  {cls}: {cnt}")
    _log(f"Triage -> {OUT_DIR / 'triage_results.parquet'}")


if __name__ == "__main__":
    run()
