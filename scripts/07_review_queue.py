"""Manual review queue processor.

Reads review_queue.parquet (proteins needing curator review) and writes an
annotated TSV for manual inspection. Column names match the aggregator output
schema (uniprot_acc, inferred_tier_a, confidence_score, etc.).

Run via:
    docker run --rm \\
        -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "git config --global --add safe.directory /pkg && \\
                 git config --global --add safe.directory /genome-atlas && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --quiet && \\
                 python scripts/07_review_queue.py"

Expected output:
  /data/labels/review_queue/review_queue_annotated.tsv
  /data/labels/review_queue/review_queue_summary.json

Column names in annotated TSV (matching aggregator schema):
  uniprot_acc, inferred_tier_a, inferred_tier_b, confidence_score,
  contradiction_flag, composite_architecture, n_sources,
  curator_decision (blank - for human), curator_notes (blank - for human)
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

REVIEW_DIR = Path("/data/labels/review_queue")
IN_PARQUET = REVIEW_DIR / "review_queue.parquet"
OUT_TSV = REVIEW_DIR / "review_queue_annotated.tsv"
OUT_JSON = REVIEW_DIR / "review_queue_summary.json"

# Column order for curator TSV (using aggregator's actual column names)
CURATOR_COLS = [
    "uniprot_acc",
    "inferred_tier_a",
    "inferred_tier_b",
    "confidence_score",
    "contradiction_flag",
    "composite_architecture",
    "n_sources",
    "conflict_notes",
    "curator_decision",    # blank - for human to fill in
    "curator_notes",       # blank - for human to fill in
]


def run() -> None:
    if not IN_PARQUET.exists():
        print(f"No review queue found at {IN_PARQUET}. Nothing to annotate.")
        return

    df = pd.read_parquet(IN_PARQUET)
    print(f"Review queue: {len(df)} proteins")

    # Add blank columns for curator input
    df["curator_decision"] = ""
    df["curator_notes"] = ""

    # Conflict notes: flag proteins with contradiction
    if "contradiction_flag" in df.columns:
        df["conflict_notes"] = df.apply(
            lambda r: f"contradiction (conf={r.get('confidence_score', 0):.2f})"
            if r.get("contradiction_flag") else "",
            axis=1,
        )
    else:
        df["conflict_notes"] = ""

    # Sort by confidence ascending (review hardest cases first)
    if "confidence_score" in df.columns:
        df = df.sort_values("confidence_score")

    # Output only columns that exist
    out_cols = [c for c in CURATOR_COLS if c in df.columns]
    df[out_cols].to_csv(OUT_TSV, sep="\t", index=False)

    # Summary JSON
    tier_a_col = "inferred_tier_a"
    comp_col   = "composite_architecture"
    summary = {
        "total_for_review": len(df),
        "by_inferred_tier_a": df[tier_a_col].value_counts().to_dict()
        if tier_a_col in df.columns else {},
        "composite_flagged": int(df[comp_col].sum())
        if comp_col in df.columns else 0,
        "contradictions": int(df["contradiction_flag"].sum())
        if "contradiction_flag" in df.columns else 0,
        "output_tsv": str(OUT_TSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Review queue annotation complete ===")
    print(f"Tier A distribution in review queue:")
    if tier_a_col in df.columns:
        print(df[tier_a_col].value_counts().to_string())
    print(f"\nAnnotated TSV -> {OUT_TSV}")
    print(f"Summary JSON  -> {OUT_JSON}")
    print(f"\nOpen {OUT_TSV} in a spreadsheet editor and fill in")
    print("  'curator_decision' (one of: DSB_NUCLEASE / DSB_FREE_TRANSEST_RECOMBINASE /")
    print("                       TRANSPOSASE / DISCARD / leave blank to keep predicted)")
    print("  'curator_notes' (free text justification)")
    print("  then run scripts/08_ingest_curator_decisions.py.")


if __name__ == "__main__":
    run()
