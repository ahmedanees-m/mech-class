"""InterPro clan inheritance for mechanism prediction.

Pfam families belong to InterPro clans; the clan encodes a mechanism inheritance
pattern (e.g., RNase H clan CL0219 → two-metal-ion phosphodiester chemistry).
Evidence weight: 0.5 (clan-level inference; lower than direct database curation).

IMPORTANT CAVEAT: CL0219 (RNase H-like) is shared between hydrolytic nucleases
(DSB_NUCLEASE) AND IS110 transesterases (DSB_FREE_TRANSEST_RECOMBINASE). This
clan-level inference is intentionally assigned DSB_NUCLEASE here; the aggregator
resolves the IS110 conflict using Pfam PF02371 (Tnp domain), TnPedia, and the
foundational systems anchor label for IS621.
"""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path

import pandas as pd
import requests
import yaml
from rich.progress import track

INTERPRO_API = "https://www.ebi.ac.uk/interpro/api/entry/pfam"

CLAN_MECHANISM: dict[str, str] = {
    "CL0219": "DSB_NUCLEASE",  # RNase H-like (RuvC, IS110 N-terminus)
    "CL0237": "DSB_NUCLEASE",  # HNH endonuclease
    "CL0184": "DSB_FREE_TRANSEST_RECOMBINASE",  # Phage integrase / lambda Int
    "CL0407": "DSB_FREE_TRANSEST_RECOMBINASE",  # Resolvase / serine recombinase
    "CL0029": "TRANSPOSASE",  # rve / DDE integrase core
}


def _load_pfam_whitelist() -> list[dict]:
    raw = yaml.safe_load(pkg_files("genome_atlas").joinpath("data/pfam_whitelist.yaml").read_text())
    return raw["domains"]


def fetch_pfam_clan(pfam_acc: str) -> str | None:
    """Return the Pfam clan accession for a given Pfam family, or None.

    InterPro v4 API structure (verified 2026-05):
      data["metadata"]["set_info"]["accession"]  — clan accession (e.g. "CL0219")
      data["metadata"]["set_info"] may be None for families with no clan.
    """
    r = requests.get(f"{INTERPRO_API}/{pfam_acc}/", timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    set_info = (data.get("metadata") or {}).get("set_info")
    if not set_info:
        return None
    return set_info.get("accession")


def main(output: Path = Path("/data/labels/evidence/interpro.parquet")) -> None:
    domains = _load_pfam_whitelist()
    rows: list[dict] = []
    for d in track(domains, description="InterPro clan queries"):
        clan = fetch_pfam_clan(d["accession"])
        rows.append(
            {
                "source": "InterPro",
                "pfam_acc": d["accession"],
                "pfam_name": d["name"],
                "clan_acc": clan,
                "inferred_tier_a": CLAN_MECHANISM.get(clan or "", "UNKNOWN"),
                "evidence_weight": 0.5,
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} InterPro clan evidence rows -> {output}")
    print(df.groupby("inferred_tier_a")["pfam_acc"].count().to_string())


if __name__ == "__main__":
    main()
