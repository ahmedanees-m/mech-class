"""Pfam whitelist v1.2.0 from genome-atlas Paper 1 — as evidence source.

The 18 primary Pfam families from Paper 1's corrected whitelist each carry a
verified mechanism_bucket annotation. This provides Pfam-level evidence (weight 0.6)
for any protein whose sequence hits one of these families.
"""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path

import pandas as pd
import yaml


def main(output: Path = Path("/data/labels/evidence/pfam_whitelist.parquet")) -> None:
    raw = yaml.safe_load(pkg_files("genome_atlas").joinpath("data/pfam_whitelist.yaml").read_text())
    domains = raw["domains"]
    version = raw.get("version", "1.2.0")

    rows = []
    for d in domains:
        rows.append(
            {
                "source": f"Pfam_whitelist_v{version}",
                "pfam_acc": d["accession"],
                "pfam_name": d["name"],
                "inferred_tier_a": d["mechanism_bucket"],
                "evidence_weight": 0.6,
                "example_systems": d.get("example_systems", []),
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} Pfam whitelist rows -> {output}")
    print(df.groupby("inferred_tier_a")["pfam_acc"].count().to_string())


if __name__ == "__main__":
    main()
