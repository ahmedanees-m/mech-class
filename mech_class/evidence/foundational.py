"""Foundational systems anchor labels from Paper 1 (genome-atlas v0.6.0).

These 16 systems are the gold-standard anchors for the training set:
all mechanism_bucket values were verified in Paper 1 (v1.2.0 audit).
Evidence weight: 1.0 (highest — directly verified in Paper 1 execution).

IS621 bridge recombinase is the critical composite case:
  mechanism_bucket: DSB_FREE_TRANSEST_RECOMBINASE (corrected from serine_recombinase)
  composite_architecture: True (RuvC-fold + serine Tnp domain)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from importlib.resources import files as pkg_files

# System name → (tier_b_id, composite_flag)
# Keys match label_taxonomy.yaml v1.0.0 examples and CRISPRCasdb system names.
# Lookup is done via _normalise_key() to tolerate spaces vs underscores, and
# parentheses vs underscores (e.g. "CAST-I-F (evoCAST)" == "CAST-I-F_evoCAST").
SYSTEM_TO_TIER_B: dict[str, tuple[str, bool]] = {
    "SpCas9":                    ("N1_CRISPR_Cas",                  False),
    "Cas12a":                    ("N1_CRISPR_Cas",                  False),
    "Cas12f":                    ("N1_CRISPR_Cas",                  False),
    "CAST-V-K":                  ("B2_CAST_Integrase",              True),
    "CAST-I-F_evoCAST":          ("B2_CAST_Integrase",              True),  # underscore form
    "CAST-I-F (evoCAST)":        ("B2_CAST_Integrase",              True),  # parens form
    "IS621_bridge_recombinase":  ("B3_Programmable_Recombinase",    True),
    "SpuFz1_Fanzor":             ("N2_Fanzor_OMEGA",                False),
    "SpuFz1_V4":                 ("N2_Fanzor_OMEGA",                False),  # variant name in YAML
    "MmeFz2_Fanzor":             ("N2_Fanzor_OMEGA",                False),
    "enNlovFz2_Fanzor":          ("N2_Fanzor_OMEGA",                False),
    "enNlovFz2":                 ("N2_Fanzor_OMEGA",                False),  # actual YAML name
    "TnsABC_CAST":               ("B2_CAST_Integrase",              True),   # CAST TnsABC complex
    "Cre_recombinase":           ("B1_Site_Specific_Recombinase",   False),
    "Bxb1_integrase":            ("B1_Site_Specific_Recombinase",   False),
    "lambda_Int":                ("B4_Tyrosine_Recombinase",        False),
    "Tn5_transposase":           ("T1_DDE_Transposase",             False),
    "PE2_prime_editor":          ("N4_Editor_Fusion",               True),
    "eePASSIGE":                 ("B1_Site_Specific_Recombinase",   False),
    "Sleeping_Beauty":           ("T1_DDE_Transposase",             False),
}

# Normalised lookup: strip parens/spaces → underscores for fuzzy key matching.
_NORMALISED = {k.replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_"): v
               for k, v in SYSTEM_TO_TIER_B.items()}


def _lookup_tier_b(system_name: str) -> tuple[str, bool]:
    """Return (tier_b, composite) for system_name, tolerating name formatting variants."""
    result = SYSTEM_TO_TIER_B.get(system_name)
    if result:
        return result
    norm = system_name.replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_")
    return _NORMALISED.get(norm, ("UNKNOWN", False))


def main(output: Path = Path("/data/labels/evidence/foundational.parquet")) -> None:
    raw = yaml.safe_load(
        pkg_files("genome_atlas").joinpath("data/foundational_systems.yaml").read_text()
    )
    systems = raw["systems"]

    rows: list[dict] = []
    unmatched = []
    for s in systems:
        tier_b, composite = _lookup_tier_b(s["name"])
        if tier_b == "UNKNOWN":
            unmatched.append(s["name"])
        for prot_acc in s.get("proteins", []):
            rows.append({
                "source": "Foundational_systems_v0.6.0",
                "uniprot_acc": prot_acc,
                "system_name": s["name"],
                "inferred_tier_a": s["mechanism_bucket"],
                "inferred_tier_b": tier_b,
                "composite_architecture": composite,
                "evidence_weight": 1.0,
            })

    if unmatched:
        print(f"  WARNING: {len(unmatched)} systems have no tier_b mapping: {unmatched}")

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression="zstd")
    print(f"Wrote {len(df):,} foundational system rows -> {output}")
    print(df.groupby("inferred_tier_a").size().to_string())
    # Verify IS621 composite case
    is621 = df[df["system_name"] == "IS621_bridge_recombinase"]
    if len(is621):
        row = is621.iloc[0]
        assert row["inferred_tier_a"] == "DSB_FREE_TRANSEST_RECOMBINASE", (
            "IS621 must be DSB_FREE not DSB_NUCLEASE — composite case!"
        )
        assert bool(row["composite_architecture"]) is True, (
            "IS621 composite_architecture must be True!"
        )
        print("IS621 composite case verified: DSB_FREE_TRANSEST_RECOMBINASE, composite=True")


if __name__ == "__main__":
    main()
