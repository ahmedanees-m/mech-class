"""M-CSA (Mechanism and Catalytic Site Atlas) REST client.

API docs: https://www.ebi.ac.uk/thornton-srv/m-csa/api/
~1,003 hand-curated enzyme entries with step-by-step mechanism descriptions
and catalytic residue assignments linked to PDB and UniProt.
Evidence weight: 1.0 (highest — hand-curated primary mechanism data).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from rich.progress import track

MCSA_API = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"

# EC prefix → Tier-A inference for DNA-modifying enzyme classes.
# Only EC classes that appear in our study domain. Entries not matching any
# prefix are left as UNKNOWN and will be filtered by the aggregator.
EC_TO_TIER_A: list[tuple[str, str]] = [
    ("3.1.21", "DSB_NUCLEASE"),   # endodeoxyribonucleases (Cas9, Cas12, restriction)
    ("3.1.22", "DSB_NUCLEASE"),   # exodeoxyribonucleases
    ("3.1.30", "DSB_NUCLEASE"),   # endo/exo phosphodiesterases (generic)
    ("3.1.4",  "DSB_NUCLEASE"),   # phosphoric diester hydrolases
    ("3.1.11", "DSB_NUCLEASE"),   # exodeoxyribonucleases on ss DNA
    ("3.1.13", "DSB_NUCLEASE"),   # exoribonucleases (RNA-guided)
]

# Mechanism text keywords for recombinase/transposase disambiguation.
# Applied when EC is ambiguous (2.7.7.*) or absent.
TRANSEST_KEYWORDS = ("transesterif", "tyrosine nucleophile", "serine nucleophile",
                     "phosphotyrosine", "phosphoserine covalent")
TRANSPOSASE_KEYWORDS = ("DDE", "transposase", "cut-and-paste")


def _infer_tier_a(ec: str, mechanism_text: str) -> str:
    """Infer Tier-A class from EC number and mechanism description."""
    for prefix, tier_a in EC_TO_TIER_A:
        if ec.startswith(prefix):
            return tier_a
    mech_lower = mechanism_text.lower()
    if any(k.lower() in mech_lower for k in TRANSEST_KEYWORDS):
        return "DSB_FREE_TRANSEST_RECOMBINASE"
    if any(k.lower() in mech_lower for k in TRANSPOSASE_KEYWORDS):
        return "TRANSPOSASE"
    return "UNKNOWN"


def fetch_all_entries(rate_limit_s: float = 0.5) -> list[dict]:
    """Page through all M-CSA entries (~1,003 total)."""
    entries = []
    url = f"{MCSA_API}/entries/"
    while url:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        entries.extend(data["results"])
        url = data.get("next")
        time.sleep(rate_limit_s)
    return entries


def parse_entry(e: dict) -> list[dict]:
    """Flatten one M-CSA entry into per-protein rows.

    M-CSA v2 API structure (verified 2026-05):
      entry["protein"]["sequences"]  — list of UniProt homologs
      entry["reaction"]["mechanisms"][0]["mechanism_text"]  — mechanism description
      entry["reaction"]["ec_number"]  — EC number
    """
    rows = []
    reaction = e.get("reaction") or {}
    mechanisms = reaction.get("mechanisms") or []
    mechanism = (mechanisms[0].get("mechanism_text") or "") if mechanisms else ""
    ec = reaction.get("ec_number") or ""
    protein = e.get("protein") or {}
    sequences = protein.get("sequences") or []
    # Collect PDB codes from sequence entries
    pdb_refs = [s["pdb_id"] for s in sequences if s.get("pdb_id")]

    tier_a = _infer_tier_a(ec, mechanism)
    for prot in sequences:
        rows.append({
            "source": "M-CSA",
            "mcsa_id": e.get("mcsa_id"),
            "uniprot_acc": prot.get("uniprot_id"),
            "ec_number": ec,
            "mechanism_text": mechanism[:1000],
            "inferred_tier_a": tier_a,
            "pdb_codes": pdb_refs,
            "evidence_weight": 1.0,
            "is_reference": prot.get("is_reference", False),
        })
    return rows


def main(output: Path = Path("/data/labels/evidence/mcsa.parquet")) -> None:
    print("Fetching M-CSA entries...")
    entries = fetch_all_entries()
    print(f"  {len(entries):,} entries fetched")

    rows: list[dict] = []
    for e in track(entries, description="Parsing M-CSA"):
        rows.extend(parse_entry(e))

    df = pd.DataFrame(rows)
    df = df[df["uniprot_acc"].notna() & (df["uniprot_acc"] != "")]
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} M-CSA evidence rows -> {output}")
    print(f"Distinct UniProt accessions: {df['uniprot_acc'].nunique():,}")


if __name__ == "__main__":
    main()
