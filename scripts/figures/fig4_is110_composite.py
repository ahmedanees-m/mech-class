"""Figure 4 - IS110 composite case: domain architecture and reclassification.

Panel A: Pfam domain architecture schematic (PF01548 + PF02371 -> IS110 composite)
Panel B: Confidence score comparison - IS110 proteins predicted as DSB_FREE vs DSB_NUCLEASE
         before and after composite override

Output: /data/figures/fig4_is110_composite.pdf  +  .png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

TRIAGE_PATH = Path("/data/results/ruvc_fold_triage.parquet")
OUT_DIR = Path("/data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _panel_a(ax: plt.Axes) -> None:
    """Domain architecture schematic for IS110 composite case."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("A. IS110 domain architecture", fontsize=10, loc="left", pad=4)

    # Protein backbone
    ax.plot([1, 9], [1.5, 1.5], color="#888888", lw=3, solid_capstyle="round", zorder=1)

    # PF01548: RuvC-DEDD (N-terminal)
    ruvc = mpatches.FancyBboxPatch((1.0, 1.05), 3.2, 0.9, boxstyle="round,pad=0.1",
                                    fc="#D62728", ec="none", alpha=0.85, zorder=2)
    ax.add_patch(ruvc)
    ax.text(2.6, 1.50, "PF01548\nRuvC-DEDD\n(RNase H-like, CL0219)", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")

    # PF02371: Ser-Tnp (C-terminal)
    ser = mpatches.FancyBboxPatch((4.8, 1.05), 3.4, 0.9, boxstyle="round,pad=0.1",
                                   fc="#1F77B4", ec="none", alpha=0.85, zorder=2)
    ax.add_patch(ser)
    ax.text(6.5, 1.50, "PF02371\nSer-Tnp C-terminal\n(IS110 recombinase)", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold")

    # Arrow and label for composite detection
    ax.annotate("", xy=(5.0, 2.6), xytext=(5.0, 2.1),
                arrowprops=dict(arrowstyle="->", color="#2CA02C", lw=2))
    ax.text(5.0, 2.75, "Composite detection: BOTH domains present\n"
            "-> DSB_FREE_TRANSEST_RECOMBINASE (overrides CL0219 -> DSB_NUCLEASE)",
            ha="center", va="center", fontsize=8, color="#2CA02C", fontweight="bold")

    ax.text(2.6, 0.7, "CL0219 clan\n(often mis-assigned\nto DSB_NUCLEASE)",
            ha="center", va="top", fontsize=7, color="#D62728", style="italic")
    ax.text(6.5, 0.7, "Serine recombinase\nmechanism\n(DSB-free)",
            ha="center", va="top", fontsize=7, color="#1F77B4", style="italic")


def _panel_b(ax: plt.Axes) -> None:
    """Confidence distribution for IS110 proteins before/after composite override."""
    ax.set_title("B. IS110 reclassification confidence", fontsize=10, loc="left", pad=4)

    if TRIAGE_PATH.exists():
        df = pd.read_parquet(TRIAGE_PATH)
        is110 = df[df["composite"] == True]
        if len(is110) > 0:
            ax.hist(is110["tier_a_confidence"], bins=15, color="#2CA02C", alpha=0.75,
                    label=f"IS110 composite (n={len(is110)})\n-> DSB_FREE_TRANSEST_RECOMBINASE")
            ax.axvline(0.60, color="#D62728", linestyle="--", linewidth=1.2,
                       label="Min confidence threshold (0.60)")
            ax.set_xlabel("Tier-A confidence score", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            return

    # Synthetic illustration if data not available
    rng = np.random.default_rng(42)
    vals = rng.beta(6, 2, size=30) * 0.4 + 0.60
    ax.hist(vals, bins=12, color="#2CA02C", alpha=0.75, label="IS110 composite (illustrative)")
    ax.axvline(0.60, color="#D62728", linestyle="--", linewidth=1.2, label="Min threshold (0.60)")
    ax.set_xlabel("Tier-A confidence score", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.text(0.5, 0.5, "Illustrative\n(run pipeline first)", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="gray", alpha=0.5)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _panel_a(axes[0])
    _panel_b(axes[1])
    fig.suptitle("IS110 Composite Case: Domain Architecture and Reclassification",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig4_is110_composite.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
