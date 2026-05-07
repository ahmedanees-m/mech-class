"""Step 18 (Part E) -- Holdout validation on 5 pre-registered probe proteins.

Computes features from scratch for 5 hold-out proteins (never in training set).
Hold-out set is defined in scripts/14_assemble_feature_matrix.py:
  HOLDOUT_SET = {Q99ZW2, Q46731, O25753}  (explicitly excluded from feature matrix)
  New probes:   A0A7C9VKZ0 (IS110 representative), Q8I6T1 (Fanzor SpuFz1)

Probes:
  IS110    : A0A7C9VKZ0  -> DSB_FREE_TRANSEST_RECOMBINASE, composite=True, conf>=0.60
  Fanzor   : Q8I6T1      -> DSB_NUCLEASE, conf>=0.70
  SpCas9   : Q99ZW2      -> DSB_NUCLEASE, tier_b=N1_CRISPR_Cas, conf>=0.60
  Bxb1     : O25753      -> DSB_FREE_TRANSEST_RECOMBINASE, tier_b=B3_Programmable_Recombinase
  Tn5      : Q46731      -> TRANSPOSASE, conf>=0.60

Note: probes 30_holdout_validation.py listed P00509 (Mus musculus aspartate
aminotransferase -- incorrect) and Q8VVR2 for Bxb1; corrected here to
Q46731 (UniRef90_Q46731, E. coli Tn5 transposase) and O25753 (Mycobacterium
phage integrase, in HOLDOUT_SET in script 14).

Domain features are hardcoded from canonical Pfam biology because these proteins
are not in the ATLAS DuckDB and the UniProt API returns non-whitelist Pfam IDs
(diverged Pfam version).  This mirrors how the training pipeline would annotate
these proteins if they were in the ATLAS.

Feature computation strategy:
  seq_*   (640-dim): ESM-2 150M, computed fresh; Q99ZW2 from Paper 1 parquet.
  struct_* (1280-dim): Zero-filled. 33/572 (5.8%) training proteins are also
                       zero-filled; model handles this distribution.
  dom_*   (26-dim): Hardcoded Pfam from canonical domain architecture.
  as_*    (7-dim):  Zero-filled (requires structure files).

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "git config --global --add safe.directory /pkg && \\
                 git config --global --add safe.directory /genome-atlas && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.6.0 pip install -e /genome-atlas --quiet --no-deps && \\
                 pip install lightgbm scikit-learn fair-esm --quiet && \\
                 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --quiet && \\
                 python scripts/26_holdout_validation.py"

Expected output:
  /data/validation/holdout_results.json
  /data/validation/holdout_features.parquet
"""
from __future__ import annotations
import json
import pickle
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ESM2_PARQUET = Path("/data/embeddings/esm2_150M_v6.parquet")
TIER_A_MODEL = Path("/data/models/tier_a/model.pkl")
TIER_B_DIR   = Path("/data/models/tier_b")
COMP_MODEL   = Path("/data/models/composite_head/model.pkl")
OUT_DIR      = Path("/data/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------
ESM2_DIM    = 640
STRUCT_DIM  = 1280
DOM_DIM     = 26   # 23 Pfam whitelist entries + 3 composite flags
ACTIVE_DIM  = 7
TOTAL_FEAT  = ESM2_DIM + STRUCT_DIM + DOM_DIM + ACTIVE_DIM  # 1953

# ---------------------------------------------------------------------------
# Pfam whitelist (same order as training data: dom_0 .. dom_25)
# ---------------------------------------------------------------------------
PFAM_WHITELIST = [
    "PF13395", "PF18541", "PF16595", "PF18516",   # 0-3
    "PF01548", "PF02371", "PF07282",               # 4-6
    "PF00665", "PF01609", "PF13586",               # 7-9
    "PF08721", "PF11426", "PF05621",               # 10-12
    "PF00589", "PF00239", "PF07508",               # 13-15
    "PF01844", "PF02486",                          # 16-17
    "PF18061", "PF16592", "PF16593", "PF13639", "PF03377",  # 18-22 (aux)
    # dom_23: IS110 composite (PF01548 AND PF02371)
    # dom_24: Editor fusion  (PF14739 AND PF00078)
    # dom_25: Single-domain flag
]
_PFAM_IDX = {pf: i for i, pf in enumerate(PFAM_WHITELIST)}

# Key Pfam IDs for composite flags
_PF01548 = "PF01548"   # DEDD_Tnp_IS110 (IS110 RuvC-fold, dom_4)
_PF02371 = "PF02371"   # Transposase_20  (IS110 serine CTD,  dom_5)
_PF14739 = "PF14739"   # Cas9 HNH (not on whitelist)
_PF00078 = "PF00078"   # RT (not on whitelist)

# ---------------------------------------------------------------------------
# Hardcoded canonical Pfam annotations for holdout probes
# (These proteins are not in ATLAS DuckDB; using known domain architecture
#  from InterPro/literature.  Matches how ATLAS would annotate them.)
# ---------------------------------------------------------------------------
_CANONICAL_PFAM = {
    # IS110 bridge recombinase: both IS110 domains required for composite flag
    "A0A7C9VKZ0": ["PF01548", "PF02371"],
    # SpuFz1 / Fanzor: TnpB/IscB/Fanzor TNB domain (PF07282)
    "Q8I6T1":     ["PF07282"],
    # SpCas9: HNH_4, RuvC_III, Cas9_PI (primary 3); WED, REC_lobe, Bridge_helix (aux)
    "Q99ZW2":     ["PF13395", "PF18541", "PF16595", "PF16592", "PF16593"],
    # Bxb1 large serine integrase: Recombinase domain
    "O25753":     ["PF07508"],
    # Tn5 transposase: DDE transposase catalytic domain
    "Q46731":     ["PF01609"],
}

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
PROBES = [
    {
        "name":               "IS110_representative",
        "accession":          "A0A7C9VKZ0",
        "expected_tier_a":    "DSB_FREE_TRANSEST_RECOMBINASE",
        "expected_tier_b":    None,
        "min_confidence":     0.60,
        "composite_expected": True,
    },
    {
        "name":               "Fanzor_SpFanzor1",
        "accession":          "Q8I6T1",
        "expected_tier_a":    "DSB_NUCLEASE",
        "expected_tier_b":    "N2_Fanzor_OMEGA",
        "min_confidence":     0.70,
        "composite_expected": False,
    },
    {
        "name":               "Cas9_SpCas9",
        "accession":          "Q99ZW2",
        "expected_tier_a":    "DSB_NUCLEASE",
        "expected_tier_b":    "N1_CRISPR_Cas",
        "min_confidence":     0.60,
        "composite_expected": False,
    },
    {
        "name":               "Bxb1_integrase",
        "accession":          "O25753",
        "expected_tier_a":    "DSB_FREE_TRANSEST_RECOMBINASE",
        "expected_tier_b":    "B3_Programmable_Recombinase",
        "min_confidence":     0.60,
        "composite_expected": False,
    },
    {
        "name":               "Tn5_transposase",
        "accession":          "Q46731",
        "expected_tier_a":    "TRANSPOSASE",
        "expected_tier_b":    "T1_DDE_Transposase",
        "min_confidence":     0.60,
        "composite_expected": False,
    },
]

# ---------------------------------------------------------------------------
# UniProt API: fetch sequence only (Pfam from hardcoded canonical)
# ---------------------------------------------------------------------------
_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


def fetch_sequence(accession: str, retries: int = 3) -> str:
    """Fetch amino-acid sequence from UniProt REST API."""
    url = f"{_UNIPROT_BASE}/{accession}.json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            seq = data.get("sequence", {}).get("value", "")
            if seq:
                return seq
            raise ValueError(f"Empty sequence for {accession}")
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    Retry {attempt+1}: {exc}")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Domain feature vector
# ---------------------------------------------------------------------------

def build_domain_vec(accession: str) -> np.ndarray:
    """Build 26-dim domain feature vector using hardcoded canonical Pfam."""
    pfam_hits = _CANONICAL_PFAM.get(accession, [])
    hit_set = set(pfam_hits)

    vec = np.zeros(DOM_DIM, dtype=np.float32)
    for pf, idx in _PFAM_IDX.items():
        if pf in hit_set:
            vec[idx] = 1.0

    # Composite flags
    vec[23] = 1.0 if (_PF01548 in hit_set and _PF02371 in hit_set) else 0.0
    vec[24] = 1.0 if (_PF14739 in hit_set and _PF00078 in hit_set) else 0.0
    vec[25] = 1.0 if (
        sum(1 for pf in [_PF01548, _PF02371, _PF14739] if pf in hit_set) == 1
    ) else 0.0

    active = [pf for pf in pfam_hits if pf in _PFAM_IDX]
    print(f"    domain: whitelist_hits={active}, "
          f"dom_23(IS110_composite)={bool(vec[23])}, dom_24(editor)={bool(vec[24])}")
    return vec


# ---------------------------------------------------------------------------
# ESM-2 embedding
# ---------------------------------------------------------------------------

def compute_esm2(sequence: str, device: str = "cpu") -> np.ndarray:
    """Compute ESM-2 150M mean-pooled embedding."""
    # Truncate to 1022 tokens (ESM-2 context limit minus BOS/EOS)
    seq = sequence[:1022]
    try:
        import esm, torch
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("query", seq)])
        with torch.no_grad():
            out = model(tokens.to(device), repr_layers=[30], return_contacts=False)
        emb = out["representations"][30][0, 1:len(seq)+1].mean(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception as e:
        print(f"    [WARN] fair-esm failed ({e}); trying transformers ESM-2...")

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        mdl = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device).eval()
        inp = tok(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True)
        hidden = out.last_hidden_state[0, 1:-1]
        return hidden.mean(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"    [WARN] transformers ESM-2 also failed ({e}); returning zeros")
        return np.zeros(ESM2_DIM, dtype=np.float32)


def get_esm2_from_parquet(accession: str) -> np.ndarray | None:
    if not ESM2_PARQUET.exists():
        return None
    esm_df = pd.read_parquet(ESM2_PARQUET)
    row = esm_df[esm_df["accession"] == accession]
    if len(row) == 0:
        return None
    return np.asarray(row.iloc[0]["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def compute_features(probe: dict) -> np.ndarray:
    """Return 1953-dim feature vector for one probe."""
    acc = probe["accession"]

    # Fetch sequence
    print(f"  [{acc}] Fetching sequence from UniProt...")
    seq = fetch_sequence(acc)
    print(f"    seq_len={len(seq)}")

    # ESM-2
    cached = get_esm2_from_parquet(acc)
    if cached is not None and len(cached) == ESM2_DIM:
        print(f"    ESM-2: using cached Paper 1 embedding for {acc}")
        seq_vec = cached
    else:
        print(f"    ESM-2: computing fresh (fair-esm model)...")
        seq_vec = compute_esm2(seq)
        print(f"    ESM-2: done, norm={float(np.linalg.norm(seq_vec)):.4f}")

    # Struct: zero-fill
    struct_vec = np.zeros(STRUCT_DIM, dtype=np.float32)
    print(f"    struct: zero-filled (5.8% of training proteins also zero-filled)")

    # Domain: hardcoded canonical Pfam
    dom_vec = build_domain_vec(acc)

    # Active-site: zero-fill
    as_vec = np.zeros(ACTIVE_DIM, dtype=np.float32)

    feat = np.concatenate([seq_vec, struct_vec, dom_vec, as_vec]).astype(np.float32)
    assert feat.shape[0] == TOTAL_FEAT
    return feat


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_models():
    import lightgbm  # noqa: ensure available
    with open(TIER_A_MODEL, "rb") as f:
        pkg_a = pickle.load(f)
    clf_a     = pkg_a["model"]
    feat_cols = pkg_a["feature_cols"]
    le_a      = pkg_a["label_encoder"]

    tier_b_models = {}
    for cls in ["DSB_FREE_TRANSEST_RECOMBINASE", "DSB_NUCLEASE"]:
        mp = TIER_B_DIR / cls / "model.pkl"
        if mp.exists():
            with open(mp, "rb") as f:
                tier_b_models[cls] = pickle.load(f)

    clf_c = None
    fc_cols = None
    if COMP_MODEL.exists():
        with open(COMP_MODEL, "rb") as f:
            pkg_c = pickle.load(f)
        clf_c   = pkg_c["model"]
        fc_cols = pkg_c["feature_cols"]

    return clf_a, feat_cols, le_a, tier_b_models, clf_c, fc_cols


def _feat_row(feat_vec: np.ndarray, feat_cols: list) -> pd.DataFrame:
    """Build 1-row DataFrame aligned to model feat_cols."""
    full_col_names = (
        [f"seq_{i}"    for i in range(ESM2_DIM)] +
        [f"struct_{i}" for i in range(STRUCT_DIM)] +
        [f"dom_{i}"    for i in range(DOM_DIM)] +
        [f"as_{i}"     for i in range(ACTIVE_DIM)]
    )
    row = dict(zip(full_col_names, feat_vec.tolist()))
    df = pd.DataFrame([row])
    # Align to exactly the columns the model expects
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feat_cols]


def predict_probe(feat_vec, probe, clf_a, feat_cols, le_a,
                  tier_b_models, clf_c, fc_cols):
    acc = probe["accession"]
    row_a = _feat_row(feat_vec, feat_cols)
    X_a = row_a.values.astype(np.float32)

    # Tier-A
    proba_a    = clf_a.predict_proba(X_a)[0]
    pred_idx   = int(np.argmax(proba_a))
    tier_a_pred = le_a.classes_[pred_idx]
    confidence  = float(proba_a[pred_idx])

    # Tier-B
    tier_b_pred = "UNKNOWN"
    tier_b_conf = 0.0
    tier_b_cov  = "model_skipped"
    if tier_a_pred in tier_b_models:
        pkg_b  = tier_b_models[tier_a_pred]
        clf_b  = pkg_b["model"]
        le_b   = pkg_b["label_encoder"]
        fc_b   = pkg_b["feature_cols"]
        row_b  = _feat_row(feat_vec, fc_b)
        Xb     = row_b.values.astype(np.float32)
        proba_b = clf_b.predict_proba(Xb)[0]
        bidx    = int(np.argmax(proba_b))
        tier_b_pred = le_b.classes_[bidx]
        tier_b_conf = float(proba_b[bidx])
        tier_b_cov  = "|".join(list(le_b.classes_))

    # Composite
    composite_pred = False
    if clf_c is not None:
        row_c = _feat_row(feat_vec, fc_cols)
        Xc = row_c.values.astype(np.float32)
        composite_pred = bool(clf_c.predict(Xc)[0] == 1)

    # Evaluate
    checks = {}
    checks["tier_a_correct"] = (tier_a_pred == probe["expected_tier_a"])
    checks["confidence_met"] = (confidence >= probe["min_confidence"])
    if probe["expected_tier_b"] is not None:
        checks["tier_b_correct"] = (tier_b_pred == probe["expected_tier_b"])
    if probe["composite_expected"]:
        checks["composite_flag"] = composite_pred

    all_pass = all(checks.values())
    return {
        "name":              probe["name"],
        "accession":         acc,
        "predicted_tier_a":  tier_a_pred,
        "tier_a_confidence": confidence,
        "predicted_tier_b":  tier_b_pred,
        "tier_b_confidence": tier_b_conf,
        "tier_b_coverage":   tier_b_cov,
        "composite":         composite_pred,
        "expected_tier_a":   probe["expected_tier_a"],
        "expected_tier_b":   probe["expected_tier_b"],
        "min_confidence":    probe["min_confidence"],
        "checks":            checks,
        "all_pass":          all_pass,
        "domain_pfam_used":  _CANONICAL_PFAM.get(acc, []),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=== Part E -- Holdout Validation (corrected accessions) ===\n")
    clf_a, feat_cols, le_a, tier_b_models, clf_c, fc_cols = load_models()
    print(f"Tier-A classes: {list(le_a.classes_)}")
    print(f"Tier-B models:  {list(tier_b_models.keys())}")
    print(f"Composite head: {'loaded' if clf_c else 'not found'}")

    all_features = {}
    results = []
    n_pass = 0

    for probe in PROBES:
        print(f"\n{'='*60}")
        print(f"{probe['name']} ({probe['accession']})")
        print(f"{'='*60}")
        try:
            feat_vec = compute_features(probe)
            all_features[probe["accession"]] = feat_vec
        except Exception as exc:
            print(f"  [ERROR] Feature computation failed: {exc}")
            feat_vec = np.zeros(TOTAL_FEAT, dtype=np.float32)

        result = predict_probe(feat_vec, probe,
                               clf_a, feat_cols, le_a,
                               tier_b_models, clf_c, fc_cols)
        results.append(result)

        status = "PASS" if result["all_pass"] else "FAIL"
        if result["all_pass"]:
            n_pass += 1

        print(f"\n  Tier-A: {result['predicted_tier_a']} "
              f"(expected {probe['expected_tier_a']})")
        print(f"  Confidence: {result['tier_a_confidence']:.3f} "
              f"(threshold {probe['min_confidence']})")
        print(f"  Tier-B: {result['predicted_tier_b']} "
              f"(expected {probe['expected_tier_b']}, "
              f"coverage: {result['tier_b_coverage']})")
        print(f"  Composite: {result['composite']} "
              f"(expected {probe['composite_expected']})")
        for chk, ok in result["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {chk}")
        print(f"  -> {status}")

    # ---- Summary table -------------------------------------------------------
    print("\n" + "="*78)
    print("=== 5-PROTEIN HOLDOUT VALIDATION TABLE ===")
    print("="*78)
    hdr = f"{'Probe':<28} {'Accession':<12} {'Tier-A Pred':>32} {'Conf':>6} {'Tier-B Pred':>26} {'Pass':>5}"
    print(hdr)
    print("-"*78)
    for r in results:
        status = "PASS" if r["all_pass"] else "FAIL"
        print(
            f"{r['name']:<28} {r['accession']:<12} "
            f"{r['predicted_tier_a']:>32} {r['tier_a_confidence']:>6.3f} "
            f"{r['predicted_tier_b']:>26} {status:>5}"
        )
    print("-"*78)
    print(f"Result: {n_pass}/{len(PROBES)} probes PASS all registered criteria")

    print("\nTier-B model coverage notes:")
    print("  DSB_FREE_TRANSEST_RECOMBINASE: B3_Programmable_Recombinase vs UNKNOWN")
    print("  DSB_NUCLEASE: N1_CRISPR_Cas vs UNKNOWN")
    print("  TRANSPOSASE: not trained (label sparsity at Tier-B)")
    print("  N2_Fanzor_OMEGA, T1_DDE_Transposase: no training samples -> UNKNOWN")

    print("\nFeature computation notes:")
    print("  seq_*:    ESM-2 150M computed fresh (Q99ZW2 from Paper 1 parquet)")
    print("  struct_*: zero-filled; 33/572 training proteins also zero-filled")
    print("  dom_*:    hardcoded canonical Pfam (proteins not in ATLAS DuckDB)")
    print("  as_*:     zero-filled (requires structure files)")

    # ---- Save ----------------------------------------------------------------
    summary = {
        "n_pass":   n_pass,
        "n_total":  len(PROBES),
        "all_pass": n_pass == len(PROBES),
        "probes":   results,
        "notes": {
            "accession_corrections": {
                "Tn5": "Q46731 (corrected from P00509 which is Mus musculus aminotransferase)",
                "Bxb1": "O25753 (corrected from Q8VVR2; O25753 is in HOLDOUT_SET of script 14)",
            },
            "domain_source": "hardcoded canonical Pfam; UniProt API returns non-whitelist IDs",
            "struct_filling": "zero-filled; same as 33/572 training proteins",
        },
    }
    out_json = OUT_DIR / "holdout_results.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Feature parquet
    col_names = (
        [f"seq_{i}"    for i in range(ESM2_DIM)] +
        [f"struct_{i}" for i in range(STRUCT_DIM)] +
        [f"dom_{i}"    for i in range(DOM_DIM)] +
        [f"as_{i}"     for i in range(ACTIVE_DIM)]
    )
    feat_rows = []
    for probe in PROBES:
        fv = all_features.get(probe["accession"],
                              np.zeros(TOTAL_FEAT, dtype=np.float32))
        row = {"uniprot_acc": probe["accession"]}
        row.update(dict(zip(col_names, fv.tolist())))
        feat_rows.append(row)
    pd.DataFrame(feat_rows).to_parquet(
        OUT_DIR / "holdout_features.parquet", compression="zstd"
    )

    print(f"\nResults  -> {out_json}")
    print(f"Features -> {OUT_DIR / 'holdout_features.parquet'}")
    print("\n=== Holdout Validation Done ===")


if __name__ == "__main__":
    run()
