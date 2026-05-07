"""Step 6b — Ingest curator decisions from review queue TSV (Week 3).

After a human has filled in curator_decision and curator_notes columns in
review_queue_annotated.tsv, this script merges decisions back into
mechanism_labels_raw.parquet and writes mechanism_labels_final.parquet.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/08_ingest_curator_decisions.py"

Expected output: /data/labels/mechanism_labels_final.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

REVIEW_TSV = Path("/data/labels/review_queue/review_queue_annotated.tsv")
RAW_LABELS = Path("/data/labels/mechanism_labels_raw.parquet")
FINAL_LABELS = Path("/data/labels/mechanism_labels_final.parquet")

VALID_TIER_A = {
    "DSB_NUCLEASE",
    "DSB_FREE_TRANSEST_RECOMBINASE",
    "TRANSPOSASE",
}
VALID_ACTIONS = VALID_TIER_A | {"DISCARD"}

# Pre-registered hold-out proteins — MUST NOT appear in training labels.
# These are excluded BEFORE final export, regardless of curator decisions.
#   Q99ZW2 — SpCas9 (Streptococcus pyogenes Cas9)
#   Q46731 — Tn5 transposase
#   O25753 — Bxb1 integrase (CRISPRCasdb)
# IS621 and SpuFz1 are not in ATLAS (foundational_systems.yaml entries are empty),
# so they will not appear in labels and need no explicit exclusion here.
HOLDOUT_PROTEINS: frozenset[str] = frozenset({
    "Q99ZW2",   # SpCas9
    "Q46731",   # Tn5
    "O25753",   # Bxb1
})


def _validate_decisions(review: pd.DataFrame) -> list[str]:
    """Validate curator TSV.  Blank = keep automated prediction (allowed)."""
    errors = []
    # Only validate non-blank rows
    filled = review[review["curator_decision"].str.strip() != ""]
    invalid = ~filled["curator_decision"].isin(VALID_ACTIONS)
    if invalid.any():
        bad = filled.loc[invalid, "curator_decision"].unique().tolist()
        errors.append(
            f"Unknown curator_decision values: {bad}. "
            f"Must be one of {VALID_ACTIONS} or blank (keep prediction)."
        )
    return errors


def run() -> None:
    if not REVIEW_TSV.exists():
        print(f"No annotated TSV at {REVIEW_TSV}. Run 07_review_queue.py first.")
        sys.exit(1)

    if not RAW_LABELS.exists():
        print(f"No raw labels at {RAW_LABELS}. Run 06_aggregate_evidence.py first.")
        sys.exit(1)

    review = pd.read_csv(REVIEW_TSV, sep="\t", dtype=str)
    review["curator_decision"] = review["curator_decision"].fillna("").str.strip()
    review["curator_notes"]    = review["curator_notes"].fillna("").str.strip()

    errors = _validate_decisions(review)
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(1)

    raw = pd.read_parquet(RAW_LABELS)

    # Build override maps (uniprot_acc is the key column from aggregator)
    key_col = "uniprot_acc"
    if key_col not in review.columns and "accession" in review.columns:
        review = review.rename(columns={"accession": key_col})

    override_map = review.set_index(key_col)["curator_decision"].to_dict()
    notes_map    = review.set_index(key_col)["curator_notes"].to_dict()

    def _apply(row: pd.Series) -> pd.Series:
        acc = row[key_col]
        decision = override_map.get(acc, "")
        if decision == "DISCARD":
            row["reviewer_action"] = "discard"
            row["curator_override"] = True
        elif decision in VALID_TIER_A:
            # Curator confirmed or corrected the tier_a label
            row["inferred_tier_a"] = decision
            row["reviewer_action"] = "auto_accept"
            row["curator_override"] = True
        else:
            # Blank = keep automated prediction
            row["curator_override"] = False
        row["curator_notes"] = notes_map.get(acc, "")
        return row

    final = raw.apply(_apply, axis=1)
    final = final[final["reviewer_action"] != "discard"].copy()

    # Remove pre-registered hold-out proteins from training labels
    before_holdout = len(final)
    final = final[~final[key_col].isin(HOLDOUT_PROTEINS)].copy()
    n_holdout_removed = before_holdout - len(final)
    if n_holdout_removed:
        print(f"Hold-out proteins removed: {n_holdout_removed} "
              f"({', '.join(sorted(HOLDOUT_PROTEINS & set(raw[key_col])))})")
    else:
        print("Hold-out check: none of the 3 pre-registered proteins present in labels (OK).")

    final.to_parquet(FINAL_LABELS, compression="zstd")

    n_override = int((final.get("curator_override", pd.Series(False)) == True).sum())
    n_discard  = int((raw["reviewer_action"] == "discard").sum()) if "reviewer_action" in raw.columns else 0
    print(f"\n=== Curator decisions ingested ===")
    print(f"Curator overrides applied: {n_override}")
    print(f"Discarded:                 {n_discard}")
    print(f"Final labeled set:         {len(final)}")
    print(f"Final labels -> {FINAL_LABELS}")


if __name__ == "__main__":
    run()
