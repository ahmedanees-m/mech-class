"""Figure 5 ? Holdout probe validation summary.

Shows confidence scores for 5 pre-registered probe proteins vs thresholds.

Output: /data/figures/fig5_holdout_probes.pdf  +  .png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Use corrected results (updated accessions); fall back to original path if absent
_corrected = Path("/data/validation/holdout_results_corrected.json")
HOLDOUT_JSON = _corrected if _corrected.exists() else Path("/data/validation/holdout_results.json")
OUT_DIR = Path("/data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBE_DISPLAY = {
    "IS110_representative": "IS110\n(A0A7C9VKZ0)",
    "Fanzor_SpFanzor1": "Fanzor\n(Q8I6T1)",
    "Cas9_SpCas9": "SpCas9\n(Q99ZW2)",
    "Bxb1_integrase": "Bxb1\n(Q9B086)",   # corrected from Q8VVR2
    "Tn5_transposase": "Tn5\n(Q46731)",    # corrected from P00509
}
THRESHOLDS = {
    "IS110_representative": 0.60,
    "Fanzor_SpFanzor1": 0.70,
    "Cas9_SpCas9": 0.60,
    "Bxb1_integrase": 0.60,
    "Tn5_transposase": 0.60,
}


def plot() -> None:
    if not HOLDOUT_JSON.exists():
        print(f"Holdout results not found: {HOLDOUT_JSON}")
        return

    data = json.loads(HOLDOUT_JSON.read_text())
    probes = {p["name"]: p for p in data["probes"]}

    names = list(PROBE_DISPLAY.keys())
    labels = [PROBE_DISPLAY[n] for n in names]
    confs = [probes[n]["confidence"] if n in probes else 0.0 for n in names]
    thresholds = [THRESHOLDS[n] for n in names]
    passed = [probes[n]["all_pass"] if n in probes else False for n in names]

    colors = ["#2CA02C" if p else "#D62728" for p in passed]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, confs, color=colors, width=0.5, edgecolor="white", linewidth=0.5)

    # Per-probe threshold ticks
    for i, thr in enumerate(thresholds):
        ax.plot([i - 0.3, i + 0.3], [thr, thr], color="black", lw=1.5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tier-A confidence score", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Holdout Probe Validation ? Confidence vs Pre-registered Thresholds",
                 fontsize=11)

    pass_patch = mpatches.Patch = plt.matplotlib.patches.Patch
    legend_elements = [
        plt.matplotlib.patches.Patch(fc="#2CA02C", label="PASS"),
        plt.matplotlib.patches.Patch(fc="#D62728", label="FAIL"),
        plt.Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Min threshold"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    n_pass = sum(passed)
    ax.text(0.02, 0.97, f"{n_pass}/{len(names)} probes PASS",
            transform=ax.transAxes, fontsize=10, va="top",
            color="#2CA02C" if n_pass == len(names) else "#D62728",
            fontweight="bold")

    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig5_holdout_probes.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
