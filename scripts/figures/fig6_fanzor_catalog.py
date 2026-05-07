"""Figure 6 — Fanzor/OMEGA catalog: confidence distribution and novelty map.

Panel A: Histogram of Tier-A confidence for Fanzor candidates
Panel B: Novelty score vs confidence scatter (novel = far from training set)
Panel C: Organism domain distribution of predicted Fanzors

Output: /data/figures/fig6_fanzor_catalog.pdf  +  .png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FANZOR_CATALOG = Path("/data/results/fanzor_catalog.parquet")
OUT_DIR = Path("/data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.70


def plot() -> None:
    if not FANZOR_CATALOG.exists():
        print(f"Fanzor catalog not found: {FANZOR_CATALOG}")
        return

    df = pd.read_parquet(FANZOR_CATALOG)
    fanzor = df[df["tier_b"] == "N2_Fanzor_OMEGA"].copy()

    if fanzor.empty:
        print("No Fanzor predictions found in catalog.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: confidence histogram
    ax = axes[0]
    confs = fanzor["tier_a_confidence"].values
    ax.hist(confs, bins=20, color="#9467BD", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(CONF_THRESHOLD, color="#D62728", linestyle="--", linewidth=1.2,
               label=f"Threshold ({CONF_THRESHOLD})")
    n_high = (confs >= CONF_THRESHOLD).sum()
    ax.set_xlabel("Tier-A confidence", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"A. Confidence distribution\n(n={len(fanzor)}, ≥{CONF_THRESHOLD}: {n_high})",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    # Panel B: novelty vs confidence scatter
    ax = axes[1]
    if "novelty_score" in fanzor.columns:
        nov = fanzor["novelty_score"].values
        conf = fanzor["tier_a_confidence"].values
        sc = ax.scatter(conf, nov, c=conf, cmap="RdYlGn", s=18, alpha=0.7,
                        vmin=0.5, vmax=1.0, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Confidence", shrink=0.8)
        ax.axvline(CONF_THRESHOLD, color="#D62728", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Tier-A confidence", fontsize=9)
        ax.set_ylabel("Novelty score\n(distance from training set)", fontsize=9)
        ax.set_title("B. Novelty vs confidence", fontsize=9)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "novelty_score\nnot computed", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("B. Novelty vs confidence", fontsize=9)

    # Panel C: organism domain pie
    ax = axes[2]
    if "organism_domain" in fanzor.columns:
        counts = fanzor["organism_domain"].value_counts()
        colors_pie = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B"]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.0f%%",
            colors=colors_pie[:len(counts)],
            pctdistance=0.75,
            startangle=90,
        )
        for t in texts:
            t.set_fontsize(8)
        for at in autotexts:
            at.set_fontsize(7)
        ax.set_title(f"C. Organism domain\n(n={len(fanzor)} Fanzor predictions)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "organism_domain\nnot available", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("C. Organism domain", fontsize=9)

    fig.suptitle("MECH-CLASS Fanzor/OMEGA-Nuclease Prospective Catalog",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig6_fanzor_catalog.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
