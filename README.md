# MECH-CLASS

[![CI](https://github.com/ahmedanees-m/mech-class/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedanees-m/mech-class/actions)
[![codecov](https://codecov.io/gh/ahmedanees-m/mech-class/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmedanees-m/mech-class)
[![PyPI](https://img.shields.io/pypi/v/mech-class.svg)](https://pypi.org/project/mech-class/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://mech-class.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.5.3-green)](CHANGELOG.md)
[![genome-atlas](https://img.shields.io/badge/built%20on-genome--atlas-informational)](https://github.com/ahmedanees-m/genome-atlas)

**Part of [PEN-STACK](https://github.com/ahmedanees-m) · Built on [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas)**

Mechanism classifier for programmable genome-writing enzymes. Given a protein sequence and optional Pfam domain annotations, MECH-CLASS assigns a **Tier-A mechanism class** (nuclease / recombinase / transposase), an optional **Tier-B sub-class**, and a **composite architecture flag** that correctly identifies IS110-family bridge recombinases — proteins that existing classifiers systematically mis-label.

---

## Architecture

MECH-CLASS is a two-tier LightGBM classifier trained on a 572-protein gold set with a 1953-dimensional feature vector fusing three channels: ESM-2 sequence embeddings (640-dim), Pfam domain flags (26-dim), and SaProt structure embeddings (1280-dim, zero-filled at inference unless a PDB path is supplied).

### Tier-A classification

Three mutually exclusive mechanism classes:

| Class | Chemistry | Representative enzymes |
|---|---|---|
| `DSB_NUCLEASE` | Hydrolytic cleavage — double-strand break produced | SpCas9, Cas12a, Fanzor, TnpB |
| `DSB_FREE_TRANSEST_RECOMBINASE` | Transesterification — no DSB, strand-transfer chemistry | IS110, CAST, Cre, Bxb1, λ-integrase |
| `TRANSPOSASE` | DDE cut-and-paste transposition | Tn5, IS10, Mos1 |

### Composite architecture detection

IS110-family proteins carry two catalytic domains: an N-terminal RuvC-fold DEDD domain (PF01548) and a C-terminal serine-transposase domain (PF02371). Standard classifiers collapse this to "DEDD nuclease" or "serine recombinase." MECH-CLASS predicts both:

- `tier_a = DSB_FREE_TRANSEST_RECOMBINASE` (transesterase chemistry, no DSB)
- `composite = True` (both domains present and catalytically relevant)

A **hard biochemical gate** (v0.5.2) reinforces this for out-of-distribution proteins: whenever PF01548 ∧ PF02371 are both present, Tier-A is forced to `DSB_FREE_TRANSEST_RECOMBINASE` regardless of what the ML model predicts. This corrects a failure mode where proteins absent from the ESM-2 training set land in an out-of-distribution feature region and receive a spurious `DSB_NUCLEASE` prediction.

### Feature channels

| Channel | Dim | Source |
|---|---|---|
| `F_seq` | 640 | ESM-2 150M mean-pool (reused from GENOME-ATLAS) |
| `F_domain` | 26 | 23 Pfam binary flags + IS110 composite flag + reserved + single-domain flag |
| `F_struct` | 1280 | SaProt 650M 3Di tokens (zero-filled unless `pdb_path=` supplied) |
| `F_active_site` | 7 | PDB/AlphaFold geometry (zero-filled unless PDB available) |
| **Total** | **1953** | Matches `features/feature_matrix.parquet` columns exactly |

---

## Quick Start

### Install

```bash
pip install mech-class
```

Add ESM-2 sequence embeddings for higher accuracy (optional — domain features alone are sufficient for most proteins):

```bash
pip install "mech-class[seq]"   # adds fair-esm + torch (~540 MB download on first use)
```

### Python API

```python
from mech_class.api import Predictor

# Load trained models (Zenodo download on first call; cached locally)
predictor = Predictor.load()

# Predict from sequence — Pfam lookup via UniProt REST happens automatically
pred = predictor.predict_from_sequence(
    accession="Q99ZW2",           # SpCas9
    sequence="MDKKYSIGLDIG...",
)
print(pred.tier_a)              # 'DSB_NUCLEASE'
print(pred.tier_a_confidence)   # 0.997
print(pred.composite)           # False

# Supply pre-computed Pfam hits to skip the UniProt lookup (recommended for batch use)
is110 = predictor.predict_from_sequence(
    accession="A0A7C9VKZ0",      # IS110 bridge recombinase
    sequence="...",
    pfam_hits=["PF01548", "PF02371"],
)
print(is110.tier_a)             # 'DSB_FREE_TRANSEST_RECOMBINASE'
print(is110.composite)          # True
print(is110.composite_prob)     # 0.999

# Batch prediction from FASTA
results = predictor.predict_from_fasta("candidates.fasta")
for r in results:
    print(r.summary())          # "A0A7C9VKZ0: DSB_FREE_TRANSEST_RECOMBINASE (conf=0.997) [COMPOSITE]"
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
| Tier-A accuracy | 0.989 | — |
| Composite head FP rate | **0%** (biochemical gate v0.5.1) | — |
| IS110 reclassification | 31,870 / 31,870 (99.9%) | — |

### OOD holdout probes (v0.5.3)

Six pre-registered out-of-distribution probes, none seen during training:

| Protein | UniProt | Expected Tier-A | Conf threshold | Gate |
|---|---|---|---|---|
| IS110 bridge recombinase | A0A7C9VKZ0 | DSB_FREE_TRANSEST_RECOMBINASE | ≥ 0.60 | — |
| Fanzor SpFanzor1 | Q8I6T1 | DSB_NUCLEASE | ≥ 0.70 | — |
| SpCas9 | Q99ZW2 | DSB_NUCLEASE | ≥ 0.60 | — |
| Bxb1 integrase | Q9B086 | DSB_FREE_TRANSEST_RECOMBINASE | ≥ 0.60 | — |
| Tn5 transposase | Q46731 | TRANSPOSASE | ≥ 0.60 | — |
| **ISCro4/IS622** (OOD gate) | **D2TGM5** | DSB_FREE_TRANSEST_RECOMBINASE | **≥ 0.90** | gate override ✓ |

D2TGM5 (ISCro4/IS622, *Citrobacter rodentium*) is the canonical gate probe: it is absent from the ESM-2 training embeddings, so the ML model predicts `DSB_NUCLEASE` P≈0.57. The IS110 gate overrides this to `DSB_FREE_TRANSEST_RECOMBINASE` with `tier_a_gate_override=True` and confidence floored at 0.90 (Pelea et al. 2026 *Science*; Perry et al. 2025 *bioRxiv*).

Known limitation: SpCas9 fires `composite=True` (P=0.753, FP). The composite head over-fires for proteins with ≥ 4 whitelist Pfam domains and no negative training examples in that regime. Documented in `MODEL_CARD.md`.

---

## Connection to GENOME-ATLAS

MECH-CLASS is Paper 2 of the PEN-STACK series; it is built directly on top of [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas) (Paper 1):

- **ESM-2 embeddings** (`F_seq`) are the same 640-dim ESM-2 150M mean-pool embeddings generated for the GENOME-ATLAS knowledge graph. They are reused as-is — no additional embedding computation is needed.
- **Label evidence** — one of eight evidence sources (`atlas_domain.py`) queries the GENOME-ATLAS DuckDB database for mechanism annotations linked to Pfam domain nodes via `HAS_DOMAIN` edges.
- **Pfam whitelist** — the 23-entry `PFAM_WHITELIST` is derived from the GENOME-ATLAS `Domain` node catalogue, filtered to mechanism-discriminating families.
- **Version pin** — `genome-atlas>=0.7.1,<0.8.0` (v0.7.1 restores `SIMILAR_TO` / `HAS_RNA` / `PART_OF` edges via `graph_view='full'` and adds ISCro4/D2TGM5 to the atlas).

---

## Key Findings

- IS110-family bridge recombinases are systematically mis-classified as `DSB_NUCLEASE` by standard domain-based classifiers. MECH-CLASS corrects 31,870/31,870 (99.9%) using the composite head + hard gate.
- Domain features alone achieve Tier-A macro-F1 ≈ 0.94. ESM-2 embeddings push this to 0.9862. Structure embeddings (SaProt) contribute < 1% on the current gold set.
- Graph topology (Node2Vec on the GENOME-ATLAS knowledge graph) achieves AUROC 0.9890 on domain-link prediction — higher than sequence-based GNNs — confirming that enzyme family relationships are primarily structural, not sequence-driven.
- The IS110 hard gate is necessary for out-of-distribution inference: without it, novel IS110-family proteins not in the ESM-2 training corpus (e.g. ISCro4/D2TGM5) receive DSB_NUCLEASE predictions with P≈0.57 (incorrect).

---

## Directory Structure

```
mech-class/
├── mech_class/                   # Python package (pip install mech-class)
│   ├── api.py                    # Predictor — main public entry point
│   ├── cli.py                    # CLI (mech-class predict)
│   ├── data/                     # Taxonomy YAML, active-site residue definitions
│   ├── evidence/                 # 8 evidence scrapers + weighted-vote aggregator
│   ├── features/                 # F_seq · F_domain · F_struct · F_active_site
│   ├── models/                   # LightGBMClassifier · CompositeHead · MLPBaseline
│   └── utils/                    # pLDDT quality filter
│
├── scripts/                      # Numbered pipeline (run on VM in order)
│   ├── 01–08  Evidence ingestion (M-CSA, Rhea, UniProt, InterPro, TnPedia…)
│   ├── 10–14  Feature computation (ESM-2, SaProt, Pfam, assembly → 1953-dim)
│   ├── 20–25  Training + ablation + bootstrap CIs
│   ├── 26–30  Holdout validation
│   ├── 40–41  Fanzor/TnpB catalog prediction (2,463 candidates)
│   ├── 50     10-probe end-to-end smoke test
│   └── figures/  Figure generation scripts (fig1–fig6)
│
├── tests/
│   ├── unit/         81 tests — aggregator, API helpers, domain features, models
│   ├── integration/  24 tests — end-to-end pipeline + Predictor API (VM-gated probes skip in CI)
│   └── regression/   VM-gated holdout + macro-F1 drift guard (skipped in CI)
│
├── docs/                         # Sphinx + Furo documentation → readthedocs.io
├── containers/structure/         # Docker image for SaProt + Foldseek (VM-only)
├── data/                         # Curator review decisions (tracked in git)
├── results/                      # Pre-computed summary JSONs
├── holdout_set.yaml              # 6 OOD probe definitions (v0.5.3)
├── MODEL_CARD.md                 # Performance, limitations, intended use
├── LABEL_PROVENANCE.md           # Gold-set label provenance and curation log
├── VALIDATION.md                 # Pre-registered success criteria and results
└── CITATION.cff                  # Machine-readable citation
```

---

## Trained Models & Data

Model artifacts are distributed via Zenodo (not bundled in this repository). `Predictor.load()` downloads them automatically on first use.

**Zenodo deposit:** DOI pending peer review — `_download_from_zenodo()` raises `RuntimeError` until the deposit is live.

| Artifact | Description | Size |
|---|---|---|
| `models/tier_a/model.pkl` | Tier-A 3-class LightGBM; macro-F1 = 0.9862 | 981 KB |
| `models/composite_head/model.pkl` | Binary IS110 composite head | 259 KB |
| `models/tier_b/*/model.pkl` | Per-class Tier-B sub-classifiers | 350–361 KB each |
| `features/feature_matrix.parquet` | 1953-dim training matrix (572 proteins) | 6.6 MB |
| `features/esm2_150M_v6.parquet` | Pre-computed ESM-2 150M embeddings | 26 MB |
| `catalogs/fanzor_candidates.parquet` | 2,463 Fanzor/TnpB predictions | 370 KB |
| `catalogs/is110_triage.parquet` | 31,871 IS110-family predictions | 455 KB |

---

## Running Tests

```bash
pip install -e ".[dev]"

# Unit + integration tests (no model files required — runs fully in CI)
pytest tests/unit/ tests/integration/ -v

# Full suite including VM-gated regression tests
pytest tests/ -v    # requires /data/models/ on VM
```

**105 unit + integration tests pass in CI** (67% coverage). Regression tests skip gracefully when model files are absent.

---

## Reproducibility

All training scripts require the VM environment (Docker, 64 GB RAM, NVIDIA GPU for SaProt). Scripts are numbered and designed to run sequentially:

```bash
# Evidence ingestion
python scripts/01_pull_mcsa.py && ... && python scripts/08_ingest_curator_decisions.py

# Feature computation (requires pen-stack/structure Docker image)
python scripts/10_compute_esm2_embeddings.py && python scripts/14_assemble_feature_matrix.py

# Training
python scripts/20_train_tier_a.py      # 5-fold stratified CV, seed=42
python scripts/22_train_composite_head.py
python scripts/24_bootstrap_cis.py    # 1000× bootstrap CIs

# Holdout validation
python scripts/30_holdout_validation.py
```

---

## Citation

If you use MECH-CLASS, please cite both papers:

```bibtex
@article{ahmed2026mechclass,
  title   = {MECH-CLASS: Structure-aware mechanism classification for programmable genome-writing enzymes},
  author  = {Ahmed, Anees},
  journal = {Briefings in Bioinformatics},
  year    = {2026},
  note    = {in submission}
}

@article{ahmed2026genomeatlas,
  title   = {GENOME-ATLAS: A unified knowledge graph for programmable genome-writing enzymes},
  author  = {Ahmed, Anees},
  journal = {Nucleic Acids Research},
  year    = {2026},
  note    = {in submission}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
