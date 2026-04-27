"""TnCentral / TnPedia evidence for IS-family and transposon systems.

TnPedia (https://tncentral.ncc.unesp.br/TnPedia/) provides the canonical reference
for IS element families and mechanisms, including the IS110 mechanism page which is
the key source for the composite DEDD+serine architecture.

IMPORTANT (verified 2026-05): TnCentral provides HTML pages only - there is no
REST API. Evidence is derived entirely from:
  1. The curated IS_FAMILY_MECHANISM table below (hand-verified from primary literature).
  2. ISfinder parquet from GENOME-ATLAS (/data/processed/isfinder.parquet), if present.

Evidence weight: 0.7 for IS-family annotations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.progress import track

# IS-family -> mechanism mapping (from TnPedia, curated 2026-04)
IS_FAMILY_MECHANISM: dict[str, tuple[str, str, bool]] = {
    # family: (tier_a, tier_b, composite)
    "IS110": ("DSB_FREE_TRANSEST_RECOMBINASE", "B3_Programmable_Recombinase", True),
    "IS1111": ("DSB_FREE_TRANSEST_RECOMBINASE", "B3_Programmable_Recombinase", True),
    "IS3": ("TRANSPOSASE", "T1_DDE_Transposase", False),
    "IS4": ("TRANSPOSASE", "T1_DDE_Transposase", False),
    "IS5": ("TRANSPOSASE", "T1_DDE_Transposase", False),
    "IS10": ("TRANSPOSASE", "T1_DDE_Transposase", False),
    "IS50": ("TRANSPOSASE", "T1_DDE_Transposase", False),
    "IS200": ("DSB_NUCLEASE", "N2_Fanzor_OMEGA", False),  # TnpB/Fanzor ancestor
    "IS605": ("DSB_NUCLEASE", "N2_Fanzor_OMEGA", False),
    "IS630": ("DSB_FREE_TRANSEST_RECOMBINASE", "B4_Tyrosine_Recombinase", False),
    "IS982": ("DSB_FREE_TRANSEST_RECOMBINASE", "B4_Tyrosine_Recombinase", False),
}

# Composite architecture note for IS110
IS110_COMPOSITE_NOTE = (
    "IS110 family: RuvC-fold N-terminal DEDD domain (PF01548) + "
    "serine Tnp C-terminal domain (PF02371). "
    "DEDD catalytic triad coordinates Mg2+ for transesterification (not hydrolysis). "
    "Reference: Hiraizumi et al. 2024 Nature; Vaysset et al. 2025 Nat Microbiol."
)


def main(
    isfinder_path: Path = Path("/data/processed/isfinder.parquet"),
    output: Path = Path("/data/labels/evidence/tnpedia.parquet"),
) -> None:
    """Build TnPedia evidence rows from ISfinder data + curated IS_FAMILY_MECHANISM table."""
    rows: list[dict] = []

    # Load ISfinder data (from GENOME-ATLAS)
    if isfinder_path.exists():
        isf = pd.read_parquet(isfinder_path)
        print(f"ISfinder: {len(isf):,} entries")
        for _, r in track(isf.iterrows(), total=len(isf), description="ISfinder"):
            family = r.get("is_family", "")
            # Match family to IS_FAMILY_MECHANISM
            matched_family = None
            for fam in IS_FAMILY_MECHANISM:
                if fam in str(family):
                    matched_family = fam
                    break
            if not matched_family:
                continue
            tier_a, tier_b, composite = IS_FAMILY_MECHANISM[matched_family]
            rows.append(
                {
                    "source": "TnPedia_ISfinder",
                    "uniprot_acc": r.get("uniprot_acc"),
                    "is_family": family,
                    "matched_family": matched_family,
                    "inferred_tier_a": tier_a,
                    "inferred_tier_b": tier_b,
                    "composite_architecture": composite,
                    "composite_note": IS110_COMPOSITE_NOTE if composite else "",
                    "evidence_weight": 0.7,
                }
            )
    else:
        print(f"ISfinder Parquet not found at {isfinder_path}; skipping ISfinder rows.")

    # Also add curated anchor rows for IS110 and IS1111 from the canonical literature
    for family, (tier_a, tier_b, composite) in IS_FAMILY_MECHANISM.items():
        rows.append(
            {
                "source": "TnPedia_curated",
                "uniprot_acc": None,  # family-level, not protein-level
                "is_family": family,
                "matched_family": family,
                "inferred_tier_a": tier_a,
                "inferred_tier_b": tier_b,
                "composite_architecture": composite,
                "composite_note": IS110_COMPOSITE_NOTE if composite else "",
                "evidence_weight": 0.7,
            }
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} TnPedia evidence rows -> {output}")
    print(df.groupby("inferred_tier_a").size().to_string())


if __name__ == "__main__":
    main()
