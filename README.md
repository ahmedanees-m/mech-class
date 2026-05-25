# MECH-CLASS

[![CI](https://github.com/ahmedanees-m/mech-class/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedanees-m/mech-class/actions)
[![codecov](https://codecov.io/gh/ahmedanees-m/mech-class/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmedanees-m/mech-class)
[![PyPI](https://img.shields.io/pypi/v/mech-class.svg)](https://pypi.org/project/mech-class/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://mech-class.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.5.4-green)](CHANGELOG.md)
[![genome-atlas](https://img.shields.io/badge/built%20on-genome--atlas-informational)](https://github.com/ahmedanees-m/genome-atlas)

**Part of [PEN-STACK](https://github.com/ahmedanees-m) · Built on [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas)**

## What is MECH-CLASS?

Every programmable genome-writing enzyme works by one of three fundamentally different chemistries. A Cas9 cuts both DNA strands (double-strand break). A Cre recombinase rearranges DNA without any break. A Tn5 transposase uses a cut-and-paste mechanism to move DNA around the genome. These differences matter enormously for safety, editing outcomes, and which cell types an enzyme can be used in - yet no existing tool could reliably classify mechanism from sequence alone, especially for novel proteins outside the training distribution.

**MECH-CLASS** solves this. It predicts the editing mechanism of any programmable genome-writing enzyme directly from its protein sequence, using a two-tier LightGBM classifier trained on 572 curated proteins. Given a sequence (or UniProt accession), it returns three outputs:

- **Tier-A class** - the broad mechanism: `DSB_NUCLEASE` (e.g. Cas9, Fanzor), `DSB_FREE_TRANSEST_RECOMBINASE` (e.g. IS110, Cre, Bxb1), or `TRANSPOSASE` (e.g. Tn5). This answers *"how does this enzyme cut or move DNA?"*
- **Tier-B sub-class** - a finer functional label within each Tier-A group (e.g. `N1_CRISPR_Cas`, `B3_Programmable_Recombinase`). Useful for distinguishing, say, a Cas12 from a Cas9, or a serine recombinase from a tyrosine recombinase.
- **Composite architecture flag** - a binary signal (`composite=True/False`) that identifies proteins carrying *two catalytic domains* working together. The canonical case is the IS110 family, which pairs a RuvC-fold nuclease domain (PF01548) with a serine-transposase domain (PF02371) - a combination that existing classifiers consistently collapse to the wrong class. MECH-CLASS is the first tool to detect and correctly label this architecture at proteome scale.

MECH-CLASS is built on top of [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas). It reuses ESM-2 protein embeddings computed by GENOME-ATLAS, draws domain annotations from the GENOME-ATLAS knowledge graph, and is designed to be used alongside it: **GENOME-ATLAS recommends *which* enzyme to use; MECH-CLASS explains *how* it works.**

---

## Why was MECH-CLASS built?

The immediate motivation was the IS110 family problem. IS110-family bridge recombinases carry a composite two-domain architecture (RuvC-fold DEDD + serine transposase). Standard sequence classifiers see the RuvC-fold domain and predict `DSB_NUCLEASE` - the wrong answer. These proteins do not make a double-strand break; they use transesterase chemistry. The misclassification has downstream consequences: enzymes predicted as nucleases get excluded from DSB-free delivery screens, and their safety profiles are misrepresented.

More broadly, as programmable genome-writing tools proliferate (Fanzors, TnpBs, IS-family recombinases, CAST transposases), the field needed a principled mechanism classifier that:
1. Works from sequence alone (no structure required)
2. Handles composite architectures (two catalytic domains in one polypeptide)
3. Degrades gracefully for out-of-distribution proteins (hard biochemical gate as a safety net)
4. Integrates with an existing enzyme knowledge graph ([GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas))

---

## How does MECH-CLASS work?

```
Protein sequence (+ optional UniProt accession)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE                         │
│                                                            │
│  F_seq  (640d)  ── ESM-2 150M mean-pool embedding         │
│                    (lazy-loaded singleton, CPU-only)       │
│                                                            │
│  F_domain (26d) ── 23 Pfam binary flags from whitelist    │
│                  + IS110 composite flag (dom_23)           │
│                  + single-domain flag (dom_25)             │
│                    (UniProt REST lookup or supplied list)  │
│                                                            │
│  F_struct (1280d) ── SaProt 3Di embeddings                │
│                      (zero-filled unless PDB supplied)     │
│                                                            │
│  F_active_site (7d) ── PDB geometry                       │
│                         (zero-filled unless PDB supplied)  │
└────────────────────────┬───────────────────────────────────┘
                         │  1953-dim feature vector
                         ▼
              ┌──────────────────┐
              │  Tier-A LightGBM │  →  DSB_NUCLEASE / DSB_FREE / TRANSPOSASE
              │  (3-class)       │     + confidence score
              └────────┬─────────┘
                       │
          ┌────────────┴──────────────┐
          │                           │
          ▼                           ▼
  ┌───────────────┐         ┌──────────────────────┐
  │  IS110 gate   │         │  Composite head       │
  │  PF01548 ∧   │         │  (binary LightGBM)   │
  │  PF02371 →   │         │  IS110 two-domain     │
  │  force DSB   │         │  architecture flag    │
  │  FREE        │         └──────────────────────┘
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │ Tier-B LightGBM │  →  sub-class label per Tier-A group
  │ (per-class)    │      (e.g. N1_CRISPR_Cas, B3_Recombinase)
  └───────────────┘
```

### The IS110 hard gate

The most critical design decision in MECH-CLASS is the biochemical hard gate. Novel IS110-family proteins not present in ESM-2's training corpus (e.g. ISCro4/D2TGM5) land in an out-of-distribution feature region when the sequence channel is zero-filled. The LightGBM confidently predicts `DSB_NUCLEASE` (P~0.57) - the biochemically incorrect class.

The gate detects this: whenever **PF01548 (DEDD_Tnp_IS110) AND PF02371 (Transposase_20)** are both present in the same protein, Tier-A is forced to `DSB_FREE_TRANSEST_RECOMBINASE` regardless of the ML score, with `tier_a_gate_override=True` and confidence floored at 0.90. This corrected 31,870/31,870 IS110-family proteins across the full UniProt proteome.

### Feature channels

| Channel | Dim | Source | Notes |
|---|---|---|---|
| `F_seq` | 640 | ESM-2 150M mean-pool | Reused from GENOME-ATLAS; zero-filled for proteins outside ESM-2 training set |
| `F_domain` | 26 | 23 Pfam binary flags + IS110 flag + reserved + single-domain flag | UniProt REST lookup at inference |
| `F_struct` | 1280 | SaProt 650M 3Di tokens | Zero-filled unless `pdb_path=` supplied |
| `F_active_site` | 7 | PDB/AlphaFold geometry | Zero-filled unless PDB available |
| **Total** | **1953** | Matches `features/feature_matrix.parquet` exactly | |

---

## Quick Start

### Install

```bash
pip install mech-class
```

Add ESM-2 sequence embeddings for higher accuracy (optional - domain features alone are sufficient for most proteins):

```bash
pip install "mech-class[seq]"   # adds fair-esm + torch (~540 MB download on first use)
```

### Python API

```python
from mech_class.api import Predictor

# Load trained models (downloaded on first call; cached locally)
predictor = Predictor.load()

# Predict from sequence - Pfam lookup via UniProt REST happens automatically
pred = predictor.predict_from_sequence(
    accession="Q99ZW2",           # SpCas9
    sequence="MDKKYSIGLDIG...",
)
print(pred.tier_a)              # 'DSB_NUCLEASE'
print(pred.tier_a_confidence)   # 0.997
print(pred.composite)           # False
print(pred.tier_b)              # 'N1_CRISPR_Cas'

# Supply pre-computed Pfam hits to skip the UniProt lookup (recommended for batch use)
is110 = predictor.predict_from_sequence(
    accession="A0A7C9VKZ0",      # IS110 bridge recombinase
    sequence="...",
    pfam_hits=["PF01548", "PF02371"],
)
print(is110.tier_a)             # 'DSB_FREE_TRANSEST_RECOMBINASE'
print(is110.composite)          # True
print(is110.composite_prob)     # 0.999
print(is110.tier_a_gate_override)  # True  ← IS110 biochemical gate fired

# Batch prediction from FASTA
results = predictor.predict_from_fasta("candidates.fasta")
for r in results:
    print(r.summary())
    # "A0A7C9VKZ0: DSB_FREE_TRANSEST_RECOMBINASE / B2_IS110_Bridge (conf=0.997) [COMPOSITE P=0.999]"

# Batch prediction from a DataFrame
import pandas as pd
df = pd.DataFrame({
    "accession": ["Q99ZW2", "A0A7C9VKZ0"],
    "sequence":  ["MDKKYSIGLDIG...", "..."],
    "pfam_hits": [["PF13395", "PF18541"], ["PF01548", "PF02371"]],
})
results_df = predictor.predict_batch(df)
```

### CLI

```bash
# Classify all sequences in a FASTA file
mech-class predict enzymes.fasta --output predictions.parquet

# With GPU-accelerated ESM-2 embeddings
mech-class predict candidates.fasta --output out.parquet --device cuda
```

---

## Validation

### Benchmark performance (v0.5.3, 572-protein gold set)

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| Tier-A macro-F1 | **0.9862** | [0.953, 1.000] |
| Tier-A accuracy | 0.989 | - |
| Composite head FP rate | **0%** (biochemical gate v0.5.1) | - |
| IS110 reclassification | 31,870 / 31,870 (99.9%) | - |

### OOD holdout probes (v0.5.4)

Six pre-registered out-of-distribution probes, none seen during training:

| Protein | UniProt | Expected Tier-A | Conf threshold | Gate |
|---|---|---|---|---|
| IS110 bridge recombinase | A0A7C9VKZ0 | DSB_FREE_TRANSEST_RECOMBINASE | >= 0.60 | - |
| Fanzor SpFanzor1 | Q8I6T1 | DSB_NUCLEASE | >= 0.70 | - |
| SpCas9 | Q99ZW2 | DSB_NUCLEASE | >= 0.60 | - |
| Bxb1 integrase | Q9B086 | DSB_FREE_TRANSEST_RECOMBINASE | >= 0.60 | - |
| Tn5 transposase | Q46731 | TRANSPOSASE | >= 0.60 | - |
| **ISCro4** (OOD gate) | **D2TGM5** | DSB_FREE_TRANSEST_RECOMBINASE | **>= 0.90** | gate override fires |

D2TGM5 (ISCro4, *Citrobacter rodentium*; formerly "IS622" in Perry et al. 2025 *bioRxiv*) is the canonical gate probe: it is absent from the ESM-2 training embeddings, so the ML model predicts `DSB_NUCLEASE` P~0.57. The IS110 gate overrides this to `DSB_FREE_TRANSEST_RECOMBINASE` with `tier_a_gate_override=True` and confidence floored at 0.90 (Pelea et al. 2026 *Science* adz1884).

**Known limitation:** SpCas9 fires `composite=True` (P=0.753, FP). The composite head over-fires for proteins with >= 4 whitelist Pfam domains and no negative training examples in that regime. Documented in `MODEL_CARD.md`.

---

## Connection to GENOME-ATLAS

MECH-CLASS builds directly on [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas):

| Integration point | Details |
|---|---|
| **ESM-2 embeddings** (`F_seq`) | The same 640-dim ESM-2 150M mean-pool embeddings from GENOME-ATLAS. No additional embedding computation needed. |
| **Label evidence** | One of eight evidence scrapers queries the GENOME-ATLAS DuckDB knowledge graph for mechanism annotations linked to Pfam domain nodes via `HAS_DOMAIN` edges. |
| **Pfam whitelist** | The 23-entry `PFAM_WHITELIST` is derived from the GENOME-ATLAS `Domain` node catalogue, filtered to mechanism-discriminating families. |
| **Version pin** | `genome-atlas>=0.7.2,<0.8.0` - v0.7.2 adds canonical ISCro4 naming, `load_systems()` API, and alias resolution with `DeprecationWarning`. |

**Designed to be used together:** GENOME-ATLAS answers *"which enzyme should I use for this editing task?"* MECH-CLASS answers *"how does that enzyme work at the molecular level?"* Together they provide a complete picture from target selection through mechanism understanding.

---

## Key Findings

- IS110-family bridge recombinases are systematically mis-classified as `DSB_NUCLEASE` by standard domain-based classifiers. MECH-CLASS corrects 31,870/31,870 (99.9%) using the composite head + hard gate.
- Domain features alone achieve Tier-A macro-F1 ~ 0.94. ESM-2 embeddings push this to 0.9862. Structure embeddings (SaProt) contribute < 1% on the current gold set.
- Graph topology (Node2Vec on GENOME-ATLAS) achieves AUROC 0.9890 on domain-link prediction - higher than sequence-based GNNs - confirming enzyme family relationships are primarily structural, not sequence-driven.
- The IS110 hard gate is *necessary* for out-of-distribution inference: without it, novel IS110-family proteins not in the ESM-2 training corpus receive DSB_NUCLEASE predictions with P~0.57 (incorrect).
- The composite head generalises beyond IS110 to other multi-domain enzyme families, with zero false positives in the holdout set when combined with the domain gate.

---

## Directory Structure

```
mech-class/
├── mech_class/                   # Python package (pip install mech-class)
│   ├── api.py                    # Predictor - main public entry point
│   ├── cli.py                    # CLI (mech-class predict)
│   ├── data/                     # Taxonomy YAML, active-site residue definitions
│   ├── evidence/                 # 8 evidence scrapers + weighted-vote aggregator
│   ├── features/                 # F_seq · F_domain · F_struct · F_active_site
│   ├── models/                   # LightGBMClassifier · CompositeHead · MLPBaseline
│   └── utils/                    # pLDDT quality filter
│
├── scripts/                      # Numbered pipeline (run in order)
│   ├── 01-08  Evidence ingestion (M-CSA, Rhea, UniProt, InterPro, TnPedia...)
│   ├── 10-14  Feature computation (ESM-2, SaProt, Pfam, assembly -> 1953-dim)
│   ├── 20-25  Training + ablation + bootstrap CIs
│   ├── 26-30  Holdout validation
│   ├── 40-41  Fanzor/TnpB catalog prediction (2,463 candidates)
│   ├── 50     10-probe end-to-end smoke test
│   └── figures/  Figure generation scripts (fig1-fig6)
│
├── tests/
│   ├── unit/         175 tests - API, domain, seq, aggregator, models, mocked paths
│   ├── integration/   31 tests - end-to-end pipeline + Predictor API (model-gated probes skip in CI)
│   └── regression/    holdout + macro-F1 drift guard (skipped in CI without model files)
│
├── docs/                         # Sphinx + Furo documentation -> readthedocs.io
├── containers/structure/         # Docker image for SaProt + Foldseek
├── data/                         # Curator review decisions (tracked in git)
├── results/                      # Pre-computed summary JSONs
├── holdout_set.yaml              # 6 OOD probe definitions (v0.5.4)
├── MODEL_CARD.md                 # Performance, limitations, intended use
├── LABEL_PROVENANCE.md           # Gold-set label provenance and curation log
├── VALIDATION.md                 # Pre-registered success criteria and results
└── CITATION.cff                  # Machine-readable citation
```

---

## Trained Models & Data

Model artifacts are provided as raw data files (not bundled in this repository). `Predictor.load()` fetches them automatically on first use.

**Model artifacts:** provided as raw data. `_download_models()` raises `RuntimeError` until a hosting URL is configured.

| Artifact | Description | Size |
|---|---|---|
| `models/tier_a/model.pkl` | Tier-A 3-class LightGBM; macro-F1 = 0.9862 | 981 KB |
| `models/composite_head/model.pkl` | Binary IS110 composite head | 259 KB |
| `models/tier_b/*/model.pkl` | Per-class Tier-B sub-classifiers | 350-361 KB each |
| `features/feature_matrix.parquet` | 1953-dim training matrix (572 proteins) | 6.6 MB |
| `features/esm2_150M_v6.parquet` | Pre-computed ESM-2 150M embeddings | 26 MB |
| `catalogs/fanzor_candidates.parquet` | 2,463 Fanzor/TnpB predictions | 370 KB |
| `catalogs/is110_triage.parquet` | 31,871 IS110-family predictions | 455 KB |

---

## Running Tests

```bash
pip install -e ".[dev]"

# Unit + integration tests (no model files required - runs fully in CI)
pytest tests/unit/ tests/integration/ -v

# Full suite including regression tests (require model files)
pytest tests/ -v
```

**179 unit + integration tests pass in CI** (96% coverage). Regression tests skip gracefully when model files are absent.

---

## Reproducibility

The training scripts require Docker, 64 GB RAM, and an NVIDIA GPU (for SaProt). They are numbered and designed to run sequentially:

```bash
# Evidence ingestion
python scripts/01_pull_mcsa.py && ... && python scripts/08_ingest_curator_decisions.py

# Feature computation (requires the structure image in containers/structure/)
python scripts/10_compute_esm2_embeddings.py && python scripts/14_assemble_feature_matrix.py

# Training
python scripts/20_train_tier_a.py      # 5-fold stratified CV, seed=42
python scripts/22_train_composite_head.py
python scripts/24_bootstrap_cis.py    # 1000x bootstrap CIs

# Holdout validation
python scripts/30_holdout_validation.py
```

---

## Citation

If you use MECH-CLASS, please cite it via the metadata in [CITATION.cff](CITATION.cff), or:

```bibtex
@software{ahmed_mechclass,
  title   = {MECH-CLASS: mechanism classification for programmable genome-writing enzymes},
  author  = {Ahmed, Anees},
  year    = {2026},
  url     = {https://github.com/ahmedanees-m/mech-class}
}
```

---

## License

MIT - see [LICENSE](LICENSE).
