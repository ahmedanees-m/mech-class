"""Step 22a — Assemble Fanzor candidate set from UniProt (replaces atlas-only query).

Queries UniProt for all eukaryotic proteins annotated with PF18297 (IS200/IS605
TnpB family -- the OMEGA-nuclease / Fanzor domain). This is the correct Pfam
for RNA-guided Fanzor endonucleases. Filters by length (200-1500 aa) and
excludes training proteins.

Also queries PF07282 (Cas12f1-like_TNB) in Eukaryota as a secondary net.

Target: 500-3,000 candidates for downstream MECH-CLASS prediction.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        python scripts/40_assemble_fanzor_candidates.py

Expected output:
  /data/predictions/fanzor_catalog/candidates.parquet
  /data/predictions/fanzor_catalog/candidates_summary.json
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("/data/predictions/fanzor_catalog")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_LABELS = Path("/data/features/fused/feature_matrix.parquet")

# UniProt REST stream endpoint
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

FANZOR_QUERIES = [
    {
        # PF07282 (Cas12f1-like_TNB / TnpB OMEGA-nuclease) in Eukaryota.
        # This is the primary Pfam marker for eukaryotic Fanzor orthologs.
        # UniProt has ~1,333 eukaryotic entries; 904 pass the 200-1500 aa filter.
        "label":     "PF07282_eukaryota",
        "query":     "(xref:pfam-PF07282) AND (taxonomy_id:2759) AND (length:[200 TO 1500])",
        "min_len":   200,
        "max_len":   1500,
    },
    {
        # PF18297 (IS200/IS605 TnpB) all kingdoms — catches prokaryotic Fanzor relatives.
        # Only 14 eukaryotic entries but 1,604 total; include all for completeness.
        "label":     "PF18297_all_kingdoms",
        "query":     "(xref:pfam-PF18297) AND (length:[200 TO 1500])",
        "min_len":   200,
        "max_len":   1500,
    },
]

FIELDS = "accession,protein_name,organism_name,length,sequence,xref_pfam"


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _query_uniprot(query: str, label: str) -> pd.DataFrame:
    _log(f"Querying UniProt: {label}")
    _log(f"  Query: {query}")

    for attempt in range(3):
        try:
            r = requests.get(
                UNIPROT_STREAM,
                params={"format": "tsv", "query": query, "fields": FIELDS},
                timeout=600,
                stream=True,
            )
            r.raise_for_status()
            content = r.content.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(content), sep="\t")
            _log(f"  Raw results: {len(df)}")
            return df
        except Exception as exc:
            _log(f"  Attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(10)

    _log(f"  [ERROR] All attempts failed for {label}")
    return pd.DataFrame()


def run() -> None:
    # Load training accessions
    training_accs: set[str] = set()
    if TRAINING_LABELS.exists():
        fm = pd.read_parquet(TRAINING_LABELS, columns=["uniprot_acc"])
        training_accs = set(fm["uniprot_acc"].tolist())
        _log(f"Training proteins (to exclude): {len(training_accs)}")

    all_dfs = []
    for q in FANZOR_QUERIES:
        df = _query_uniprot(q["query"], q["label"])
        if df.empty:
            continue

        # Standardize column names
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "entry" in cl and "name" not in cl:
                col_map[c] = "accession"
            elif "length" in cl:
                col_map[c] = "length"
            elif "protein name" in cl or "protein_name" in cl:
                col_map[c] = "protein_name"
            elif "organism" in cl and "id" not in cl:
                col_map[c] = "organism"
            elif "sequence" in cl:
                col_map[c] = "sequence"
            elif "pfam" in cl or "xref_pfam" in cl:
                col_map[c] = "pfam_refs"
        df = df.rename(columns=col_map)

        if "accession" not in df.columns:
            _log(f"  [WARN] Cannot find accession column in {df.columns.tolist()}")
            continue

        # Length filter
        if "length" in df.columns:
            df["length"] = pd.to_numeric(df["length"], errors="coerce")
            before = len(df)
            df = df[df["length"].between(q["min_len"], q["max_len"])]
            _log(f"  After length filter ({q['min_len']}-{q['max_len']} aa): {len(df)} (was {before})")

        df["source_nomination"] = q["label"]
        all_dfs.append(df)

    if not all_dfs:
        _log("[ERROR] No candidates retrieved from any query. Check network connectivity.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Keep required columns
    keep = [c for c in ["accession", "protein_name", "organism", "length", "sequence",
                         "pfam_refs", "source_nomination"] if c in combined.columns]
    combined = combined[keep]

    # Dedup
    before = len(combined)
    combined = combined.drop_duplicates(subset=["accession"], keep="first")
    _log(f"After dedup: {len(combined)} (was {before})")

    # Exclude training
    combined = combined[~combined["accession"].isin(training_accs)].reset_index(drop=True)
    _log(f"After training exclusion: {len(combined)}")

    # Drop rows without sequence
    if "sequence" in combined.columns:
        before = len(combined)
        combined = combined[combined["sequence"].notna() & (combined["sequence"] != "")]
        _log(f"After sequence filter: {len(combined)} (was {before})")

    combined.to_parquet(OUT_DIR / "candidates.parquet", compression="zstd")
    combined.to_csv(OUT_DIR / "candidates.tsv", sep="\t", index=False)

    summary = {
        "total_candidates": len(combined),
        "source_counts": combined["source_nomination"].value_counts().to_dict() if "source_nomination" in combined.columns else {},
        "length_stats": {
            "min": int(combined["length"].min()) if "length" in combined.columns else None,
            "max": int(combined["length"].max()) if "length" in combined.columns else None,
            "mean": round(float(combined["length"].mean()), 1) if "length" in combined.columns else None,
        },
        "queries": [q["label"] for q in FANZOR_QUERIES],
        "training_excluded": len(training_accs),
    }
    (OUT_DIR / "candidates_summary.json").write_text(json.dumps(summary, indent=2))

    _log(f"\n=== Fanzor candidates assembled ===")
    _log(f"Total: {len(combined)}")
    _log(f"Saved -> {OUT_DIR / 'candidates.parquet'}")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
