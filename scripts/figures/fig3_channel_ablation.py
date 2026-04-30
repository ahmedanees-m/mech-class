"""Figure 3 - Channel ablation bar chart.

Shows macro-F1 +/- 95% CI for each ablation condition, with full model highlighted.

Output: /data/figures/fig3_channel_ablation.pdf  +  .png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ABLATION_JSON = Path("/data/models/ablation/ablation_results.json")
OUT_DIR = Path("/data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "full": "Full model\n(F_seq+F_struct+F_dom+F_as)",
    "-seq": "-F_seq",
    "-struct": "-F_struct",
    "-domain": "-F_domain",
    "-active_site": "-F_active_site",
    "seq_only": "F_seq only",
    "domain_only": "F_domain only",
}


def plot() -> None:
    if not ABLATION_JSON.exists():
        print(f"Ablation results not found: {ABLATION_JSON}")
        return

    results = json.loads(ABLATION_JSON.read_text())
    results = [r for r in results if not r.get("skipped") and r.get("bootstrap_f1") is not None]

    labels = [LABEL_MAP.get(r["label"], r["label"]) for r in results]
    f1s = np.array([r["bootstrap_f1"] for r in results])
    lo = np.array([r["ci_lower"] for r in results])
    hi = np.array([r["ci_upper"] for r in results])
    yerr_lo = f1s - lo
    yerr_hi = hi - f1s

    colors = ["#2CA02C" if r["label"] == "full" else "#AEC7E8" for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    bars = ax.bar(x, f1s, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    ax.errorbar(x, f1s, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=4, linewidth=1.2)

    # Pre-registered threshold lines
    ax.axhline(0.80, color="#D62728", linestyle="--", linewidth=1.0, label="Target F1 = 0.80")
    ax.axhline(0.70, color="#FF7F0E", linestyle=":", linewidth=1.0, label="CI lower bound = 0.70")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro-F1 (bootstrap, n=1000)", fontsize=10)
    ax.set_title("Channel Ablation Study - Tier-A Macro-F1", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig3_channel_ablation.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
