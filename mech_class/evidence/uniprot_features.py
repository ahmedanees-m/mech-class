"""UniProt SwissProt ACT_SITE and BINDING feature parser.

Pulls reviewed entries that match Paper 1's Pfam whitelist and extracts
catalytic residue annotations via the UniProt search JSON API.
Evidence weight: 0.7.

Notes on endpoint choice (verified 2026-05):
  - /stream with TSV + sequence field returns HTTP 400 (payload too large).
  - /search with JSON + pagination works reliably; sequence field removed.
  - Query format: (database:Pfam AND xref:PF00000) AND reviewed:true
"""

from __future__ import annotations

import re
import time
from importlib.resources import files as pkg_files
from pathlib import Path

import pandas as pd
import requests
import yaml
from rich.progress import track

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
RATE_LIMIT_S = 0.3
PAGE_SIZE = 500

MECHANISM_KEYWORDS: dict[str, list[str]] = {
    "DSB_NUCLEASE": [
        r"\bnuclease\b",
        r"endonuclease",
        r"DNase",
        r"phosphodiesterase",
        r"two-metal-ion catalysis",
        r"hydrolytic cleavage",
    ],
    "DSB_FREE_TRANSEST_RECOMBINASE": [
        r"recombinase",
        r"integrase",
        r"transesterification",
        r"phosphoester intermediate",
        r"covalent serine",
        r"covalent tyrosine",
        r"site-specific recombination",
        r"strand exchange",
    ],
    "TRANSPOSASE": [
        r"transposase",
        r"transposition",
        r"DDE motif",
        r"strand transfer",
    ],
}


def _load_pfam_whitelist() -> list[dict]:
    raw = yaml.safe_load(pkg_files("genome_atlas").joinpath("data/pfam_whitelist.yaml").read_text())
    return raw["domains"]


def query_pfam(pfam_acc: str) -> list[dict]:
    """Fetch reviewed UniProt entries with ACT_SITE / BINDING features for a Pfam family.

    Uses cursor-based pagination on /search to retrieve all results without
    the payload-size limits of /stream.

    Query format verified 2026-05: xref_pfam:PF00000 AND reviewed:true
    Features requested: ft_act_site, ft_binding, ft_metal (no sequence field).
    UniProt returns a 'features' array in JSON; each element has 'type' and 'description'.
    """
    # ft_metal is NOT a valid UniProt field (verified 2026-05).
    # xref_pfam is NOT a valid field — use xref:pfam-PF00000 in query instead.
    fields = "accession,protein_name,organism_name,length,ft_act_site,ft_binding"
    params = {
        "format": "json",
        "query": f"(xref:pfam-{pfam_acc}) AND (reviewed:true)",
        "fields": fields,
        "size": PAGE_SIZE,
    }
    all_results: list[dict] = []
    cursor_url: str | None = UNIPROT_SEARCH

    while cursor_url:
        if cursor_url == UNIPROT_SEARCH:
            r = requests.get(cursor_url, params=params, headers={"Accept": "application/json"}, timeout=60)  # type: ignore[arg-type]
        else:
            r = requests.get(cursor_url, headers={"Accept": "application/json"}, timeout=60)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data.get("results", []))

        # Follow Link: <url>; rel="next" header for pagination
        link_header = r.headers.get("Link", "")
        cursor_url = None
        if 'rel="next"' in link_header:
            # Extract URL between < >
            import re as _re

            m = _re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
            if m:
                cursor_url = m.group(1)
        time.sleep(RATE_LIMIT_S)

    return all_results


def _extract_features(entry: dict, target_types: tuple[str, ...]) -> str:
    """Extract plain-text descriptions from UniProt JSON feature entries.

    UniProt REST API v2 JSON structure (verified 2026-05):
    - 'features' is a list of dicts, each with 'type', 'description', 'ligand'.
    - 'type' values: "Active site", "Binding site" (title case, no ft_ prefix).
    - 'description' may be empty for binding sites; 'ligand.name' has the cofactor.
    - No separate field keys per type — all features are in one 'features' list.

    target_types: tuple of strings to match against feat['type'] (case-insensitive).
    """
    parts: list[str] = []

    for feat in entry.get("features") or []:
        ftype = (feat.get("type") or "").lower()
        if any(t.lower() in ftype for t in target_types):
            desc = feat.get("description") or ""
            # For binding sites, description may be empty; use ligand name instead
            if not desc:
                ligand = feat.get("ligand") or {}
                desc = ligand.get("name") or ""
            if desc:
                parts.append(desc)

    return " | ".join(parts)


def infer_mechanism_from_features(act_site: str, binding: str, name: str) -> tuple[str, float]:
    """Pattern-match feature line text against mechanism keywords."""
    text = " ".join(filter(None, [act_site or "", binding or "", name or ""])).lower()
    scores: dict[str, int] = {}
    for mech, patterns in MECHANISM_KEYWORDS.items():
        scores[mech] = sum(1 for p in patterns if re.search(p, text))
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return ("UNKNOWN", 0.0)
    return (best, min(1.0, scores[best] / 3))


def main(output: Path = Path("/data/labels/evidence/uniprot_features.parquet")) -> None:
    domains = _load_pfam_whitelist()
    rows: list[dict] = []
    for d in track(domains, description="UniProt feature queries"):
        try:
            entries = query_pfam(d["accession"])
        except Exception as exc:
            print(f"WARN {d['accession']}: {exc}")
            continue
        for entry in entries:
            acc = entry.get("primaryAccession") or ""
            if not acc:
                continue

            # Extract protein name
            desc = entry.get("proteinDescription") or {}
            rec = desc.get("recommendedName") or {}
            protein_name = (rec.get("fullName") or {}).get("value") or ""

            # Features are in entry['features'] list; filter by 'type' field.
            # Verified types (2026-05): "Active site", "Binding site"
            act_site_text = _extract_features(entry, ("active site",))
            binding_text = _extract_features(entry, ("binding site",))

            mech, conf = infer_mechanism_from_features(
                act_site_text,
                binding_text,
                protein_name,
            )
            if mech == "UNKNOWN":
                continue
            rows.append(
                {
                    "source": "UniProt_features",
                    "uniprot_acc": acc,
                    "pfam_acc": d["accession"],
                    "inferred_tier_a": mech,
                    "inferred_confidence": conf,
                    "act_site_text": act_site_text[:500],
                    "binding_text": binding_text[:500],
                    "evidence_weight": 0.7,
                }
            )

    df_out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output, compression="zstd")
    print(f"\nWrote {len(df_out):,} UniProt feature evidence rows -> {output}")


if __name__ == "__main__":
    main()
