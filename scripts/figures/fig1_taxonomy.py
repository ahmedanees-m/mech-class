"""Figure 1 - Mechanism taxonomy tree (Tier A -> Tier B hierarchy).

Generates a two-level dendrogram/sunburst showing the 3 Tier-A classes
and their 10 Tier-B sub-classes, with training set counts per node.

Output: /data/figures/fig1_taxonomy.pdf  +  .png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

LABELS_PATH = Path("/data/labels/mechanism_labels_final.parquet")
OUT_DIR = Path("/data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIER_A_COLORS = {
    "DSB_NUCLEASE": "#D62728",
    "DSB_FREE_TRANSEST_RECOMBINASE": "#2CA02C",
    "TRANSPOSASE": "#1F77B4",
}

TIER_B_MAP = {
    "DSB_NUCLEASE": ["N1_CRISPR_Cas", "N2_Fanzor_OMEGA", "N3_Generic_Nuclease", "N4_Editor_Fusion"],
    "DSB_FREE_TRANSEST_RECOMBINASE": [
        "B1_Site_Specific_Recombinase", "B2_CAST_Integrase",
        "B3_Programmable_Recombinase", "B4_Tyrosine_Recombinase", "B5_Serine_Recombinase",
    ],
    "TRANSPOSASE": ["T1_DDE_Transposase"],
}

TIER_B_LABELS = {
    "N1_CRISPR_Cas": "CRISPR-Cas\nnuclease",
    "N2_Fanzor_OMEGA": "Fanzor /\nOMEGA",
    "N3_Generic_Nuclease": "Generic\nnuclease",
    "N4_Editor_Fusion": "Editor\nfusion",
    "B1_Site_Specific_Recombinase": "Site-specific\nrecombinase",
    "B2_CAST_Integrase": "CAST\nintegrase",
    "B3_Programmable_Recombinase": "Programmable\nrecombinase",
    "B4_Tyrosine_Recombinase": "Tyrosine\nrecombinase",
    "B5_Serine_Recombinase": "Serine\nrecombinase",
    "T1_DDE_Transposase": "DDE\ntransposase",
}


def _load_counts() -> tuple[dict[str, int], dict[str, int]]:
    if not LABELS_PATH.exists():
        return {k: 0 for k in TIER_A_COLORS}, {k: 0 for v in TIER_B_MAP.values() for k in v}
    df = pd.read_parquet(LABELS_PATH)
    tier_a_counts = df["tier_a"].value_counts().to_dict()
    tier_b_counts = df["tier_b"].value_counts().to_dict() if "tier_b" in df.columns else {}
    return tier_a_counts, tier_b_counts


def plot() -> None:
    tier_a_counts, tier_b_counts = _load_counts()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Root node
    root_x, root_y = 1.0, 3.5
    ax.text(root_x, root_y, "DNA-modifying\nenzyme", ha="center", va="center",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888", lw=1.5))

    tier_a_positions = {
        "DSB_NUCLEASE": (3.5, 5.8),
        "DSB_FREE_TRANSEST_RECOMBINASE": (3.5, 3.5),
        "TRANSPOSASE": (3.5, 1.2),
    }
    tier_a_short = {
        "DSB_NUCLEASE": "DSB\nNuclease",
        "DSB_FREE_TRANSEST_RECOMBINASE": "DSB-Free\nRecombinase",
        "TRANSPOSASE": "Transposase",
    }

    tier_b_x = 7.5
    tier_b_slots = {
        "DSB_NUCLEASE": [6.1, 5.4, 4.7, 4.0],
        "DSB_FREE_TRANSEST_RECOMBINASE": [3.5, 3.0, 2.5, 2.0, 1.5],
        "TRANSPOSASE": [1.2],
    }

    for tier_a, (ax_x, ax_y) in tier_a_positions.items():
        color = TIER_A_COLORS[tier_a]
        n = tier_a_counts.get(tier_a, 0)
        ax.plot([root_x + 0.7, ax_x - 0.7], [root_y, ax_y], color="#AAAAAA", lw=1.2, zorder=1)
        ax.text(ax_x, ax_y, f"{tier_a_short[tier_a]}\n(n={n})", ha="center", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec=color, alpha=0.85, lw=0), color="white")

        slots = tier_b_slots[tier_a]
        tier_b_list = TIER_B_MAP[tier_a]
        for k, (tier_b, slot_y) in enumerate(zip(tier_b_list, slots)):
            n_b = tier_b_counts.get(tier_b, 0)
            label = TIER_B_LABELS.get(tier_b, tier_b)
            ax.plot([ax_x + 0.7, tier_b_x - 0.7], [ax_y, slot_y], color="#CCCCCC", lw=0.9, zorder=1)
            ax.text(tier_b_x, slot_y, f"{label}\n(n={n_b})", ha="center", va="center",
                    fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc=color, ec=color, alpha=0.45, lw=0))

    legend_patches = [
        mpatches.Patch(color=c, label=k.replace("_", " ").title())
        for k, c in TIER_A_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)

    ax.set_title("MECH-CLASS Taxonomy (Tier A -> Tier B)", fontsize=13, fontweight="bold", pad=10)

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig1_taxonomy.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    plot()
