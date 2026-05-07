"""Step 5 — Aggregate all evidence sources → mechanism_labels_raw.parquet (Week 3).

Reads all /data/labels/evidence/*.parquet files, runs weighted vote aggregation
per UniProt accession, applies IS110 composite override rule, and outputs:
  - mechanism_labels_raw.parquet: all proteins with aggregated evidence records
  - review_queue.parquet: proteins requiring manual review (confidence 0.5-0.75)

Run after scripts 01-05d are complete.

    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/06_aggregate_evidence.py"

Expected output:
  /data/labels/mechanism_labels_raw.parquet  (~150-300 proteins)
  /data/labels/review_queue/review_queue.parquet  (proteins needing manual review)
"""
from __future__ import annotations
from pathlib import Path

from mech_class.evidence.aggregator import main as aggregate_main
import pandas as pd


def run():
    # Run aggregation
    aggregate_main(
        evidence_dir=Path("/data/labels/evidence"),
        output=Path("/data/labels/mechanism_labels_raw.parquet"),
        atlas_db="/data/graphs/atlas.duckdb",
    )

    # Split into review queue
    df = pd.read_parquet("/data/labels/mechanism_labels_raw.parquet")
    review = df[df["reviewer_action"] == "manual_review"].copy()
    auto   = df[df["reviewer_action"] == "auto_accept"].copy()
    discard = df[df["reviewer_action"] == "discard"].copy()

    review.to_parquet(
        "/data/labels/review_queue/review_queue.parquet", compression="zstd"
    )

    print(f"\n=== Evidence aggregation complete ===")
    print(f"Auto-accept:   {len(auto):>4}")
    print(f"Manual review: {len(review):>4}")
    print(f"Discard:       {len(discard):>4}")
    print(f"Total:         {len(df):>4}")
    print(f"\nReview queue → /data/labels/review_queue/review_queue.parquet")
    print(f"Raw labels   → /data/labels/mechanism_labels_raw.parquet")


if __name__ == "__main__":
    run()
