"""Rhea reactions database REST client — 2-step Rhea→UniProt lookup.

Rhea API docs: https://www.rhea-db.org/help/rest-api
UniProt REST API: https://rest.uniprot.org/

Step 1: Search Rhea by keyword to retrieve Rhea reaction IDs relevant to
        DNA modification (nucleases, recombinases, transposases).
Step 2: For each Rhea reaction ID, query UniProt for reviewed proteins
        annotated with that specific Rhea reaction.

Why the redesign (verified 2026-05):
  - Rhea EC search `ec:3.1.21.-` returns 0 results (broken endpoint).
  - Rhea reaction search results contain NO UniProt xref fields.
  - The Rhea→protein link lives in UniProt, not in Rhea — must query UniProt
    with `rhea:{numeric_id}` to get catalyzing proteins.
  - `xref_rhea:RHEA:XXXXX` and `database:Rhea AND xref:{id}` both return 400.
  - Rhea search API returns IDs as strings (not integers).

Evidence weight: 0.8 (lower than M-CSA because reaction-level annotation,
not step-by-step mechanism characterization).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

RHEA_SEARCH = "https://www.rhea-db.org/rhea"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Keyword queries to search in Rhea for relevant reaction classes.
# Short, broad terms — Rhea is a small DB (~17k reactions) so false positives
# are filtered downstream by UniProt reviewed:true.
RHEA_QUERIES = [
    "endonuclease",
    "transposase",
    "recombinase",
    "integrase",
    "CRISPR",
]

# EC prefix → Tier A mapping (covers DNA-modifying enzymes).
# EC 3.1.x = phosphodiester bond hydrolases (nucleases)
# EC 2.7.7.x = nucleotidyltransferases (some repair enzymes)
EC_TO_TIER_A: list[tuple[str, str]] = [
    ("3.1.21", "DSB_NUCLEASE"),   # type I, II, III DNA restriction endonucleases
    ("3.1.22", "DSB_NUCLEASE"),   # ssDNA endonucleases
    ("3.1.30", "DSB_NUCLEASE"),   # S1 nucleases
    ("3.1.4",  "DSB_NUCLEASE"),   # phosphodiesterases
    ("3.1.11", "DSB_NUCLEASE"),   # exodeoxyribonucleases
    ("3.1.13", "DSB_NUCLEASE"),   # exoribonucleases
    ("3.1.26", "DSB_NUCLEASE"),   # ribonucleases H
    ("3.1.25", "DSB_NUCLEASE"),   # AP endonucleases / site-specific
]

TRANSEST_NAME_KEYWORDS = (
    "recombinase", "integrase", "transesterase",
    "site-specific", "strand exchange", "tyrosine recombinase",
    "serine recombinase",
)
TRANSPOSASE_NAME_KEYWORDS = (
    "transposase", "IS element", "DDE transposase",
)


def _infer_tier_a_rhea(ec: str, protein_name: str) -> str:
    """Infer Tier A from Rhea-linked protein's EC number and name."""
    for prefix, tier_a in EC_TO_TIER_A:
        if ec.startswith(prefix):
            return tier_a
    name_lower = protein_name.lower()
    if any(k.lower() in name_lower for k in TRANSPOSASE_NAME_KEYWORDS):
        return "TRANSPOSASE"
    if any(k.lower() in name_lower for k in TRANSEST_NAME_KEYWORDS):
        return "DSB_FREE_TRANSEST_RECOMBINASE"
    # Default: treat Rhea nuclease/repair reactions as DSB_NUCLEASE unless
    # name clearly indicates otherwise (conservative — filtered by confidence).
    if "nuclease" in name_lower or "endonuclease" in name_lower:
        return "DSB_NUCLEASE"
    return "UNKNOWN"

# Maximum Rhea IDs to carry forward per query (avoid unbounded UniProt calls).
MAX_RHEA_IDS = 50
MAX_UNIPROT_PER_REACTION = 200
RATE_LIMIT_S = 0.4


def search_rhea(query: str) -> list[str]:
    """Return Rhea reaction IDs (as strings) matching the keyword query."""
    try:
        r = requests.get(
            RHEA_SEARCH,
            params={"query": query, "format": "json", "limit": MAX_RHEA_IDS},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        # IDs come as strings from the Rhea API (e.g. "66592")
        return [str(rxn["id"]) for rxn in results if rxn.get("id")]
    except Exception as exc:
        print(f"  Rhea search '{query}' failed: {exc}")
        return []


def fetch_uniprot_for_rhea(rhea_id: str) -> list[dict]:
    """Return reviewed UniProt entries annotated with the given Rhea reaction ID.

    Correct UniProt REST query (verified 2026-05):
      rhea:{numeric_id} AND reviewed:true
    Other formats (`database:Rhea AND xref:{id}`, `xref_rhea:RHEA:{id}`) all
    return HTTP 400. The working format uses just the bare numeric Rhea ID.
    """
    try:
        r = requests.get(
            UNIPROT_SEARCH,
            params={
                "query": f"(rhea:{rhea_id}) AND (reviewed:true)",
                "fields": "accession,protein_name,organism_name",
                "format": "json",
                "size": MAX_UNIPROT_PER_REACTION,
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as exc:
        print(f"  UniProt query for RHEA:{rhea_id} failed: {exc}")
        return []


def parse_uniprot_entry(entry: dict, rhea_id: str) -> dict:
    acc = entry.get("primaryAccession") or ""
    # Extract EC number from nested proteinDescription structure
    desc = (entry.get("proteinDescription") or {})
    rec = desc.get("recommendedName") or {}
    ec_list_items = rec.get("ecNumbers") or []
    ec = ",".join(e.get("value", "") for e in ec_list_items)

    protein_name = ""
    desc = (entry.get("proteinDescription") or {})
    rec = desc.get("recommendedName") or {}
    full_name = rec.get("fullName") or {}
    protein_name = full_name.get("value") or ""

    tier_a = _infer_tier_a_rhea(ec, protein_name)

    return {
        "source": "Rhea",
        "rhea_id": rhea_id,
        "uniprot_acc": acc,
        "ec_number": ec,
        "protein_name": protein_name,
        "inferred_tier_a": tier_a,
        "evidence_weight": 0.8,
    }


def main(output: Path = Path("/data/labels/evidence/rhea.parquet")) -> None:
    # Step 1: collect Rhea reaction IDs from keyword searches
    all_rhea_ids: set[str] = set()
    for query in RHEA_QUERIES:
        print(f"Searching Rhea: '{query}'...")
        ids = search_rhea(query)
        print(f"  {len(ids)} reaction IDs")
        all_rhea_ids.update(ids)
        time.sleep(RATE_LIMIT_S)

    print(f"\nTotal unique Rhea reaction IDs: {len(all_rhea_ids)}")

    # Step 2: for each Rhea ID, query UniProt
    all_rows: list[dict] = []
    seen_accs: set[str] = set()
    for rhea_id in sorted(all_rhea_ids):
        entries = fetch_uniprot_for_rhea(rhea_id)
        time.sleep(RATE_LIMIT_S)
        for entry in entries:
            row = parse_uniprot_entry(entry, rhea_id)
            acc = row.get("uniprot_acc") or ""
            if acc and acc not in seen_accs:
                seen_accs.add(acc)
                all_rows.append(row)

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["source", "rhea_id", "uniprot_acc", "ec_number", "protein_name",
                 "inferred_tier_a", "evidence_weight"]
    )
    df = df[df["uniprot_acc"].notna() & (df["uniprot_acc"] != "")]
    # Drop rows where tier_a could not be inferred (no evidence value to aggregator)
    n_before = len(df)
    df = df[df["inferred_tier_a"] != "UNKNOWN"]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with UNKNOWN tier_a")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"\nWrote {len(df):,} Rhea evidence rows (deduplicated by UniProt acc) -> {output}")
    print(f"Distinct UniProt accessions: {df['uniprot_acc'].nunique():,}")


if __name__ == "__main__":
    main()
