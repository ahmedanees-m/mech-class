"""Figure 2 - Tier-A 5-fold CV confusion matrix.

Output: /data/figures/fig2_confusion_matrix.pdf  +  .png
"""
from __future__ import annotations
from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

from mech_class.models.lightgbm_clf import MechLGBMClassifier

X_PATH = Path("/data/features/X.parquet")
Y_PATH = Path("/data/features/y.parquet")
MODEL_PATH = Path("/data/models/tier_a/model.lgb")
CV_PATH = Path("/data/models/tier_a/cv_results.json")
OUT_DIR = Path("/data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["DSB_NUCLEASE", "DSB_FREE_\nTRANSEST_RECOMBINASE", "TRANSPOSASE"]
CLASS_KEYS = ["DSB_NUCLEASE", "DSB_FREE_TRANSEST_RECOMBINASE", "TRANSPOSASE"]


def plot() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    X_df = pd.read_parquet(X_PATH)
    y_df = pd.read_parquet(Y_PATH)
    merged = X_df.merge(y_df[["accession", "tier_a"]], on="accession", how="inner").dropna(subset=["tier_a"])
    feature_cols = [c for c in merged.columns if c not in ("accession", "tier_a", "tier_b", "composite_flag")]

    X = merged[feature_cols].values.astype(np.float32)
    y_true = merged["tier_a"].values

    clf = MechLGBMClassifier.load(MODEL_PATH)
    y_pred = clf.predict(X)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_KEYS, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")

    if CV_PATH.exists():
        cv = json.loads(CV_PATH.read_text())
        f1 = cv.get("macro_f1_mean", 0)
        f1_std = cv.get("macro_f1_std", 0)
        ci_lo = cv.get("bootstrap_ci_lower", 0)
        ci_hi = cv.get("bootstrap_ci_upper", 0)
        ax.set_title(
            f"Tier-A Confusion Matrix (normalized)\n"
            f"Macro-F1 = {f1:.3f} +/- {f1_std:.3f}  "
            f"[95% CI: {ci_lo:.3f}-{ci_hi:.3f}]",
            fontsize=11,
        )
    else:
        ax.set_title("Tier-A Confusion Matrix (normalized)", fontsize=11)

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig2_confusion_matrix.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
