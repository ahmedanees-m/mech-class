"""Step 17 -- Holdout validation on five pre-registered probe proteins (Week 6).

Pre-registered probes (never seen during training):
  IS110    : A0A7C9VKZ0  --> DSB_FREE_TRANSEST_RECOMBINASE  (IS110 composite)
  Fanzor   : Q8I6T1      --> DSB_NUCLEASE
  Cas9     : Q99ZW2      --> DSB_NUCLEASE
  Bxb1     : Q9B086      --> DSB_FREE_TRANSEST_RECOMBINASE  (corrected from Q8VVR2)
  Tn5      : Q46731      --> TRANSPOSASE                     (corrected from P00509)
  Cre      : P06956      --> [IN TRAINING -- excluded from OOD evaluation]

Success criteria (Tier-A only; Tier-B UNKNOWN is acceptable per label_taxonomy.yaml
because sub-class training N < 3 for TRANSPOSASE, N=39 for DSB_NUCLEASE):
  IS110    : tier_a == DSB_FREE_TRANSEST_RECOMBINASE, confidence >= 0.60, composite==True
  Fanzor   : tier_a == DSB_NUCLEASE, confidence >= 0.70
  Cas9     : tier_a == DSB_NUCLEASE, confidence >= 0.60
  Bxb1     : tier_a == DSB_FREE_TRANSEST_RECOMBINASE, confidence >= 0.60
  Tn5      : tier_a == TRANSPOSASE, confidence >= 0.60

Accession corrections (2026-05-06):
  Bxb1: Q8VVR2 (S. aureus GajA nuclease, wrong) -> Q9B086 (Mycobacteriophage Bxb1, correct)
  Tn5 : P00509 (wrong placeholder) -> Q46731 (E. coli Tn5 transposase, correct)
  See LABEL_PROVENANCE.md Data Pipeline Corrections.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install lightgbm --quiet && python scripts/30_holdout_validation.py"

Expected output:
  /data/validation/holdout_results_corrected.json
  /data/validation/holdout_results.json         (same content; for backward compat)
  /data/validation/holdout_table.txt
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

OUT_DIR      = Path("/data/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR    = Path("/data/models")
TIER_A_PATH  = MODEL_DIR / "tier_a" / "model.pkl"
COMP_PATH    = MODEL_DIR / "composite_head" / "model.pkl"
HOLDOUT_FEAT = OUT_DIR / "holdout_features.parquet"
ATLAS_DB     = Path("/data/graphs/atlas.duckdb")

# Pfam whitelist (dom_0..dom_22)
PFAM_WHITELIST = [
    "PF13395", "PF18541", "PF16595", "PF18516", "PF01548", "PF02371",
    "PF07282", "PF00665", "PF01609", "PF13586", "PF08721", "PF11426",
    "PF05621", "PF00589", "PF00239", "PF07508", "PF01844", "PF02486",
    "PF18061", "PF16592", "PF16593", "PF13639", "PF03377",
]

_ESM2_MODEL = None
_ESM2_ALPHABET = None
_ESM2_BATCH_CONVERTER = None


def _load_esm2() -> None:
    global _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_BATCH_CONVERTER
    if _ESM2_MODEL is not None:
        return
    try:
        import esm as fair_esm
        _ESM2_MODEL, _ESM2_ALPHABET = fair_esm.pretrained.esm2_t30_150M_UR50D()
        _ESM2_MODEL = _ESM2_MODEL.eval()
        _ESM2_BATCH_CONVERTER = _ESM2_ALPHABET.get_batch_converter()
        _log("ESM-2 150M loaded.")
    except Exception as exc:
        _log(f"[WARN] ESM-2 load failed: {exc}. Sequence channel will be zero-filled.")


def _embed_sequence(seq: str) -> np.ndarray:
    if _ESM2_MODEL is None:
        return np.zeros(640, dtype=np.float32)
    import torch
    try:
        seq = seq[:1022]
        batch = [("x", seq)]
        _, _, tokens = _ESM2_BATCH_CONVERTER(batch)
        with torch.no_grad():
            out = _ESM2_MODEL(tokens, repr_layers=[30])
        return out["representations"][30][0, 1:-1].mean(0).cpu().numpy().astype(np.float32)
    except Exception as exc:
        _log(f"[WARN] ESM-2 embed failed: {exc}")
        return np.zeros(640, dtype=np.float32)


def _fetch_uniprot_sequence(accession: str) -> str:
    """Fetch FASTA sequence from UniProt REST API."""
    try:
        import urllib.request
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        with urllib.request.urlopen(url, timeout=15) as resp:
            fasta = resp.read().decode("utf-8")
        lines = [l for l in fasta.strip().split("\n") if not l.startswith(">")]
        return "".join(lines).strip()
    except Exception as exc:
        _log(f"  [WARN] UniProt fetch failed for {accession}: {exc}")
        return ""


def _get_atlas_features(accession: str, feat_cols: list[str],
                        canonical_pfam: list[str] | None = None) -> np.ndarray:
    """Compute feature row for a probe not in holdout_features.parquet.

    Strategy (in order):
      1. Atlas: sequence + domain Pfam hits
      2. If not in atlas: fetch sequence from UniProt; use canonical_pfam for domain flags
      3. ESM-2 embedding from sequence (if available)
    """
    row = np.zeros(len(feat_cols), dtype=np.float32)
    col_map = {c: i for i, c in enumerate(feat_cols)}

    seq = ""
    pfam_set: set[str] = set()

    # Try atlas first
    if ATLAS_DB.exists():
        con = duckdb.connect(str(ATLAS_DB), read_only=True)
        seq_row = con.execute(
            "SELECT sequence FROM nodes_protein WHERE accession = ?", [accession]
        ).fetchone()
        if seq_row:
            seq = seq_row[0] or ""
        pfam_rows = con.execute("""
            SELECT d.accession
            FROM nodes_protein p
            JOIN edges e ON e.source_id = p.id AND e.source_type = 'Protein'
            JOIN nodes_domain d ON d.id = e.target_id AND e.target_type = 'Domain'
            WHERE p.accession = ?
        """, [accession]).fetchall()
        pfam_set = {r[0] for r in pfam_rows}
        con.close()

    # If not in atlas, try UniProt
    if not seq:
        _log(f"  [{accession}] not in atlas -- fetching from UniProt")
        seq = _fetch_uniprot_sequence(accession)

    # Use canonical_pfam as fallback domain hits when atlas has none
    if not pfam_set and canonical_pfam:
        pfam_set = set(canonical_pfam)
        _log(f"  [{accession}] using canonical_pfam fallback: {canonical_pfam}")

    # ESM-2 embedding
    if seq:
        emb = _embed_sequence(seq)
        for k in range(640):
            c = f"seq_{k}"
            if c in col_map:
                row[col_map[c]] = emb[k]

    # Domain flags
    wl_hits = []
    for wl_idx, pfam in enumerate(PFAM_WHITELIST):
        c = f"dom_{wl_idx}"
        if c in col_map and pfam in pfam_set:
            row[col_map[c]] = 1.0
            wl_hits.append(pfam)

    if "dom_23" in col_map:
        row[col_map["dom_23"]] = float("PF01548" in pfam_set and "PF02371" in pfam_set)
    if "dom_25" in col_map:
        row[col_map["dom_25"]] = float(len(wl_hits) == 1)

    _log(f"  [{accession}] fallback: seq={'yes' if seq else 'no'} ({len(seq)} aa), "
         f"pfam={sorted(pfam_set & set(PFAM_WHITELIST))}")
    return row

PROBES = [
    {
        "name":               "IS110_representative",
        "accession":          "A0A7C9VKZ0",
        "expected_tier_a":    "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_confidence":     0.60,
        "composite_expected": True,
    },
    {
        "name":               "Fanzor_SpFanzor1",
        "accession":          "Q8I6T1",
        "expected_tier_a":    "DSB_NUCLEASE",
        "min_confidence":     0.70,
        "composite_expected": False,
    },
    {
        "name":               "Cas9_SpCas9",
        "accession":          "Q99ZW2",
        "expected_tier_a":    "DSB_NUCLEASE",
        "min_confidence":     0.60,
        "composite_expected": False,
        "note":               "Composite FP (P=0.753) -- see MODEL_CARD.md",
    },
    {
        "name":               "Bxb1_integrase",
        "accession":          "Q9B086",
        "expected_tier_a":    "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_confidence":     0.60,
        "composite_expected": False,
        "canonical_pfam":     ["PF07508", "PF00239"],
        "note":               "Corrected from Q8VVR2 (S. aureus GajA nuclease, wrong). Q9B086 not in atlas -- fetched from UniProt.",
    },
    {
        "name":               "Tn5_transposase",
        "accession":          "Q46731",
        "expected_tier_a":    "TRANSPOSASE",
        "min_confidence":     0.60,
        "composite_expected": False,
        "canonical_pfam":     ["PF01609"],
        "note":               "Corrected from P00509 (wrong placeholder)",
    },
    {
        "name":               "Cre_recombinase",
        "accession":          "P06956",
        "expected_tier_a":    "DSB_FREE_TRANSEST_RECOMBINASE",
        "min_confidence":     0.60,
        "composite_expected": False,
        "in_distribution":    True,
        "note":               "IN TRAINING (row 8658, PF00589). Not OOD -- included for composite FP table (§0.5 pre-registration). Composite P=0.005 -> TN.",
    },
]

CRE_NOTE = (
    "P06956 (Cre recombinase) is IN TRAINING (row 8658, DSB_FREE / B1_Site_Specific_Recombinase, "
    "424 PF00589 proteins in training). Cannot serve as OOD holdout. Included in composite FP "
    "table as in-distribution probe per §0.5 pre-registration requirement. "
    "ACTUAL: tier_a=DSB_FREE conf=0.9999, composite=False (P=0.005) -> TN (in-distribution). "
    "Composite FP rate (4 probes incl. Cre): 1/4 = 25%. See LABEL_PROVENANCE.md."
)


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _load_models():
    with open(TIER_A_PATH, "rb") as f:
        ta = pickle.load(f)
    with open(COMP_PATH, "rb") as f:
        comp = pickle.load(f)
    return ta["model"], ta["feature_cols"], ta["label_encoder"], comp["model"], comp["feature_cols"]


def _get_feature_row(accession: str, feat_cols: list[str],
                     canonical_pfam: list[str] | None = None) -> np.ndarray:
    """Pull pre-computed features, or fall back to atlas/UniProt computation."""
    if HOLDOUT_FEAT.exists():
        df = pd.read_parquet(HOLDOUT_FEAT)
        match = df[df["uniprot_acc"] == accession]
        if not match.empty:
            col_map = {c: i for i, c in enumerate(feat_cols)}
            row = np.zeros(len(feat_cols), dtype=np.float32)
            for c in feat_cols:
                if c in match.columns:
                    row[col_map[c]] = float(match.iloc[0][c])
            return row
    _log(f"  [{accession}] not in holdout_features.parquet -- computing from atlas/UniProt")
    return _get_atlas_features(accession, feat_cols, canonical_pfam)


def run() -> None:
    lgbm_a, feat_cols, le_a, lgbm_comp, comp_feat_cols = _load_models()
    _log("Models loaded.")
    _load_esm2()  # for atlas fallback feature computation

    results = []
    n_pass = 0
    _log("\n=== Holdout Validation (corrected accessions) ===\n")

    for probe in PROBES:
        acc = probe["accession"]
        x = _get_feature_row(acc, feat_cols, probe.get("canonical_pfam")).reshape(1, -1)
        x_df = pd.DataFrame(x, columns=feat_cols)

        proba_a = lgbm_a.predict_proba(x_df)[0]
        pred_idx = int(np.argmax(proba_a))
        tier_a = le_a.inverse_transform([pred_idx])[0]
        conf = float(proba_a[pred_idx])

        x_comp = x_df[comp_feat_cols] if comp_feat_cols else x_df
        comp_proba = lgbm_comp.predict_proba(x_comp)[0]
        composite = bool(comp_proba[1] >= 0.5)

        in_dist = probe.get("in_distribution", False)

        checks = {}
        checks["tier_a_correct"] = (tier_a == probe["expected_tier_a"])
        checks["confidence_met"] = (conf >= probe["min_confidence"])
        if probe["composite_expected"]:
            checks["composite_flag"] = composite

        all_pass = all(checks.values())
        if all_pass and not in_dist:
            n_pass += 1

        status = "PASS" if all_pass else "FAIL"
        if in_dist:
            status += " (IN-DIST)"
        _log(f"{probe['name']} ({acc}): {status}")
        _log(f"  Tier-A: {tier_a} (expected {probe['expected_tier_a']})")
        _log(f"  Confidence: {conf:.3f} (min {probe['min_confidence']})")
        _log(f"  Composite P(True): {comp_proba[1]:.3f} -> {composite}")
        if probe.get("note"):
            _log(f"  Note: {probe['note']}")
        for k, v in checks.items():
            _log(f"  [{'PASS' if v else 'FAIL'}] {k}")
        _log("")

        results.append({
            "name":               probe["name"],
            "accession":          acc,
            "predicted_tier_a":   tier_a,
            "tier_a_confidence":  conf,
            "composite":          composite,
            "composite_prob":     float(comp_proba[1]),
            "expected_tier_a":    probe["expected_tier_a"],
            "min_confidence":     probe["min_confidence"],
            "checks":             checks,
            "all_pass":           all_pass,
            "in_distribution":    in_dist,
            "note":               probe.get("note", ""),
        })

    n_ood = sum(1 for p in PROBES if not p.get("in_distribution", False))
    _log(f"Result: {n_pass}/{n_ood} OOD probes PASS (Cre is in-distribution, not counted)")
    _log(f"  (Tier-B UNKNOWN for all probes -- acceptable per label_taxonomy.yaml)")
    _log(f"\nNote on Cre (P06956): {CRE_NOTE}")

    # Composite FP summary (all 4 probes: Cas9/Bxb1/Cre/Tn5 -- Cre included per §0.5)
    composite_probes = [r for r in results if r["name"] != "IS110_representative" and r["name"] != "Fanzor_SpFanzor1"]
    comp_fp = sum(1 for r in composite_probes if r["composite"])
    _log(f"\nComposite FP rate ({len(composite_probes)} probes incl. Cre): {comp_fp}/{len(composite_probes)} = {100*comp_fp/len(composite_probes):.0f}%")
    _log(f"  Pre-registered threshold: ≤10% -- {'PASS' if comp_fp/len(composite_probes) <= 0.10 else 'FAIL'}")

    summary = {
        "n_pass":    n_pass,
        "n_ood":     n_ood,
        "n_total":   len(PROBES),
        "all_pass":  n_pass == n_ood,
        "probes":    results,
        "cre_note":  CRE_NOTE,
        "composite_fp_rate": {
            "numerator":   comp_fp,
            "denominator": len(composite_probes),
            "rate":        round(comp_fp / max(len(composite_probes), 1), 4),
            "threshold":   0.10,
            "pass":        comp_fp / max(len(composite_probes), 1) <= 0.10,
            "probes":      [r["name"] for r in composite_probes],
        },
        "accession_corrections": {
            "Bxb1": "Q8VVR2 -> Q9B086 (Mycobacteriophage Bxb1 integrase)",
            "Tn5":  "P00509 -> Q46731 (E.coli Tn5 transposase)",
        },
    }

    corrected_path = OUT_DIR / "holdout_results_corrected.json"
    corrected_path.write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "holdout_results.json").write_text(json.dumps(summary, indent=2))

    header = f"{'Probe':<30} {'Accession':<14} {'Tier-A':<30} {'Conf':>6}  {'CompP':>6}  Pass"
    sep    = "-" * 92
    lines  = [header, sep]
    for r in results:
        dist_flag = " [in-dist]" if r["in_distribution"] else ""
        lines.append(
            f"{r['name']:<30} {r['accession']:<14} {r['predicted_tier_a']:<30} "
            f"{r['tier_a_confidence']:>6.3f}  {r['composite_prob']:>6.3f}  "
            f"{'PASS' if r['all_pass'] else 'FAIL'}{dist_flag}"
        )
    lines += [sep,
              f"  {n_pass}/{n_ood} OOD PASS  (Cre in-distribution, not counted in OOD total)",
              f"  Composite FP: {comp_fp}/{len(composite_probes)} = {100*comp_fp/len(composite_probes):.0f}% (threshold ≤10%: {'PASS' if comp_fp/len(composite_probes)<=0.10 else 'FAIL'})"]
    table_text = "\n".join(lines) + "\n"
    (OUT_DIR / "holdout_table.txt").write_text(table_text)
    _log(f"\n{table_text}")
    _log(f"Results -> {corrected_path}")


if __name__ == "__main__":
    run()
