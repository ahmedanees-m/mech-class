# mech-class

[![CI](https://github.com/ahmedanees-m/mech-class/workflows/CI/badge.svg)](https://github.com/ahmedanees-m/mech-class/actions)
[![codecov](https://codecov.io/gh/ahmedanees-m/mech-class/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmedanees-m/mech-class)
[![PyPI](https://img.shields.io/pypi/v/mech-class.svg)](https://pypi.org/project/mech-class/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://mech-class.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Part of [PEN-STACK](https://github.com/ahmedanees-m)**

Structure-aware mechanism classifier for DNA-modifying enzymes. Predicts catalytic mechanism across three Tier-A classes with Tier-B sub-classification, from protein sequence and structure, with **explicit handling of composite catalytic architectures** (IS110 family bridge recombinases).

Built on top of [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas) (Paper 1 of PEN-STACK).

---

## Scientific contribution

The key novel contribution is explicit handling of composite catalytic architectures. IS110 family proteins carry a RuvC-fold DEDD N-terminal domain and a serine-recombinase Tnp C-terminal domain. Existing classifiers collapse this to "DEDD nuclease" (wrong) or "serine recombinase" (incomplete). MECH-CLASS predicts:

- Tier A: `DSB_FREE_TRANSEST_RECOMBINASE` (correct — no DSB, transesterase chemistry)
- `composite_architecture: True` (RuvC-fold + serine Tnp, both domains required)

**Three Tier-A mechanism classes:**

| Class | Chemistry | Examples |
|---|---|---|
| `DSB_NUCLEASE` | Hydrolytic phosphodiester cleavage, DSB produced | Cas9, Cas12a/f, Fanzor, TnpB |
| `DSB_FREE_TRANSEST_RECOMBINASE` | Transesterification, no DSB | IS110, CAST, Cre, Bxb1, Lambda Int |
| `TRANSPOSASE` | DDE-family cut-and-paste transposition | Tn5, IS10, Mos1 |

**Performance (v0.5.3, 572-protein gold set):**

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| Tier-A macro-F1 | 0.9862 | [0.953, 1.000] |
| Composite head FP rate | 0% (biochemical gate v0.5.1) | — |
| IS110 holdout confidence | 0.997 | — |
| OOD gate probe (ISCro4/D2TGM5) | ≥0.90 (gate floor) | gate override ✓ |

---

## Install

```bash
pip install mech-class
pip install lightgbm   # required for inference
```

For sequence embedding (improves accuracy; not required for domain-feature-only predictions):

```bash
pip install fair-esm torch
```

## Quickstart

```python
from mech_class.api import Predictor

# Load trained models (downloads from Zenodo on first call)
predictor = Predictor.load()

# Predict from sequence — UniProt accession triggers Pfam lookup automatically
pred = predictor.predict_from_sequence(
    accession="Q99ZW2",           # SpCas9
    sequence="MDKKYSIGLDIGTNSVGWAVITDEYKVPS...",
)
print(pred.tier_a)              # 'DSB_NUCLEASE'
print(pred.tier_a_confidence)   # 0.997
print(pred.composite)           # False

# Supply pre-computed Pfam hits to bypass UniProt lookup (recommended for batch use)
is110_pred = predictor.predict_from_sequence(
    accession="A0A7C9VKZ0",
    sequence="...",
    pfam_hits=["PF01548", "PF02371"],   # IS110: RuvC-fold + serine Tnp
)
print(is110_pred.tier_a)              # 'DSB_FREE_TRANSEST_RECOMBINASE'
print(is110_pred.composite)           # True
print(is110_pred.composite_prob)      # 0.999
print(is110_pred.tier_a_gate_override) # False (ESM-2 available; ML correct w/o gate)

# OOD IS110 probe — no ESM-2 embedding; gate fires (D2TGM5, ISCro4/IS622)
iscro4_pred = predictor.predict_from_sequence(
    accession="D2TGM5",
    sequence="...",
    pfam_hits=["PF01548", "PF02371"],
)
print(iscro4_pred.tier_a)               # 'DSB_FREE_TRANSEST_RECOMBINASE'
print(iscro4_pred.tier_a_gate_override) # True  (gate fired; ML would have said DSB_NUCLEASE)
print(iscro4_pred.tier_a_confidence)    # >= 0.90 (gate floor)
```

## Command-line interface

```bash
# Predict mechanism for all sequences in a FASTA file
mech-class predict proteins.fasta --output predictions.parquet

# Predict with GPU-accelerated ESM-2 embeddings
mech-class predict fanzor_orthologs.fasta --output fanzor_predictions.parquet --device cuda
```

### Biochemical gate logic (v0.5.2)

MECH-CLASS uses two hard biochemical gates in `api.py` — both keyed on the same Pfam co-occurrence:

| Gate | Condition | Effect |
|---|---|---|
| **Tier-A IS110 gate** (v0.5.2) | PF01548 ∧ PF02371 in pfam_hits | Forces `tier_a = DSB_FREE_TRANSEST_RECOMBINASE`; sets `tier_a_gate_override=True` |
| **Composite gate** (v0.5.1) | PF01548 ∧ PF02371 in pfam_hits | Allows `composite=True` if ML prob ≥ 0.5; forces `composite=False` otherwise |

The Tier-A gate fixes an OOD inference failure: when ESM-2 embeddings are unavailable (domain-only path), IS110-family proteins land in an out-of-distribution feature space and the LightGBM model incorrectly predicts DSB_NUCLEASE. The biochemical gate overrides this — PF01548 ∧ PF02371 co-occurrence definitionally identifies IS110-family bridge recombinases (Hiraizumi et al. 2024 *Nature*; Vaysset et al. 2025 *Nat Microbiol*), which are always DSB_FREE_TRANSEST_RECOMBINASE.

---

## Feature channels

| Channel | Dimension | Source | Notes |
|---|---|---|---|
| `F_seq` | 640 | ESM-2 150M mean-pool | Reused from GENOME-ATLAS Paper 1; lazy-loaded singleton |
| `F_struct` | 1280 | SaProt 650M 3Di tokens | Zero-filled at inference unless `pdb_path=` provided |
| `F_domain` | 26 | Pfam binary flags + composite | dom_0..22: PFAM_WHITELIST; dom_23: IS110 composite; dom_24: reserved; dom_25: single-domain flag |
| `F_active_site` | 7 | PDB/AlphaFold geometry | Zero-filled unless PDB available; pLDDT ≥ 70 filter |
| **Total** | **1953** | | Matches `features/feature_matrix.parquet` columns exactly |

**PFAM_WHITELIST (23 entries, fixed order dom_0..dom_22):**
`PF13395 PF18541 PF16595 PF18516 PF01548 PF02371 PF07282 PF00665 PF01609 PF13586 PF08721 PF11426 PF05621 PF00589 PF00239 PF07508 PF01844 PF02486 PF18061 PF16592 PF16593 PF13639 PF03377`

---

## Repository layout

```
mech-class/
│
├── mech_class/                     Python package (installable)
│   ├── __init__.py                 Public API re-exports
│   ├── _version.py                 Version string (setuptools-scm)
│   ├── api.py                      Predictor class — main public entry point
│   ├── cli.py                      Click CLI (`mech-class predict`)
│   ├── data/
│   │   ├── active_site_residues.yaml  Catalytic residue definitions per family
│   │   ├── label_taxonomy.yaml        Tier-A / Tier-B class hierarchy
│   │   └── loader.py                  Data-loading helpers
│   ├── evidence/                   Label construction (8 evidence sources)
│   │   ├── aggregator.py           Weighted vote → EvidenceRecord; IS110 override rule
│   │   ├── mcsa.py                 M-CSA mechanistic annotations
│   │   ├── rhea.py                 Rhea reaction database
│   │   ├── uniprot_features.py     UniProt active-site / binding-site features
│   │   ├── interpro.py             InterPro clan CL0219 (DEDD fold)
│   │   ├── tnpedia.py              TnPedia / ISfinder transposase catalog
│   │   ├── crisprcasdb.py          CRISPRCasdb CRISPR effector annotations
│   │   ├── pfam_whitelist.py       Pfam-to-mechanism whitelist
│   │   ├── foundational.py         Foundational systems YAML (expert-curated)
│   │   └── atlas_domain.py         GENOME-ATLAS DuckDB domain evidence
│   ├── features/                   Feature engineering
│   │   ├── seq.py                  F_seq: ESM-2 150M embeddings (640-dim)
│   │   ├── domain.py               F_domain: Pfam binary flags + composite (26-dim)
│   │   ├── struct.py               F_struct: SaProt 650M 3Di tokens (1280-dim)
│   │   └── active_site.py          F_active_site: PDB geometry (7-dim)
│   ├── models/                     Classifier wrappers
│   │   ├── lightgbm_clf.py         LightGBMClassifier (fit/predict/macro_f1/save/load)
│   │   ├── composite_head.py       CompositeHead binary classifier (IS110 detection)
│   │   └── mlp_clf.py              MLPClassifier baseline (ablation only)
│   └── utils/
│       └── plddt.py                pLDDT-based structure quality filter
│
├── scripts/                        Numbered pipeline scripts (run in order on VM)
│   ├── 00_smoke_import.py          Package import smoke test
│   │
│   ├── 01_pull_mcsa.py             Evidence ingestion: M-CSA
│   ├── 02_pull_rhea.py             Evidence ingestion: Rhea
│   ├── 03_pull_uniprot_features.py Evidence ingestion: UniProt active-site features
│   ├── 04_pull_interpro.py         Evidence ingestion: InterPro CL0219
│   ├── 05_pull_tnpedia.py          Evidence ingestion: TnPedia / ISfinder
│   ├── 05b_pull_crisprcasdb.py     Evidence ingestion: CRISPRCasdb
│   ├── 05c_pull_pfam_whitelist.py  Evidence ingestion: Pfam whitelist
│   ├── 05d_pull_foundational.py    Evidence ingestion: Foundational systems
│   ├── 05e_pull_atlas_domains.py   Evidence ingestion: GENOME-ATLAS domains
│   ├── 06_aggregate_evidence.py    Weighted vote → mechanism_labels_raw.parquet
│   ├── 07_review_queue.py          Flag contradictions → review_queue.parquet
│   ├── 08_ingest_curator_decisions.py  Apply manual decisions → gold set
│   │
│   ├── 10_compute_esm2_embeddings.py   F_seq: ESM-2 150M (reused from Paper 1)
│   ├── 11_compute_saprot_embeddings.py F_struct: SaProt 650M 3Di tokens
│   ├── 12_compute_active_site_features.py  F_active_site: PDB geometry (7-dim)
│   ├── 13_compute_domain_features.py   F_domain: Pfam binary flags (26-dim)
│   ├── 14_assemble_feature_matrix.py   Fuse all channels → 1953-dim matrix
│   │
│   ├── 20_train_tier_a.py          Train Tier-A 3-class LightGBM (5-fold CV)
│   ├── 21_train_tier_b.py          Train per-class Tier-B LightGBM sub-classifiers
│   ├── 22_train_composite_head.py  Train binary IS110 composite head
│   ├── 23_channel_ablation.py      7-condition channel ablation study
│   ├── 24_bootstrap_cis.py         1000× bootstrap CIs on test set
│   ├── 25_train_mlp_baseline.py    MLP baseline for ablation comparison
│   ├── 26_holdout_validation.py    5-probe OOD holdout validation
│   ├── 27_spcas9_composite_check.py  SpCas9 composite FP characterisation
│   ├── 28_bxb1_saprot.py           Bxb1 SaProt structure embedding check
│   ├── 29_holdout_corrected.py     Holdout with accession corrections
│   ├── 30_holdout_validation.py    Final holdout validation (all 5 probes)
│   │
│   ├── 40_assemble_fanzor_candidates.py  Stage 0: Fanzor/TnpB candidate list
│   ├── 40_predict_fanzor_catalog.py      Stage 1: Tier-A predictions (2,463 candidates)
│   ├── 41_predict_ruvc_fold_catalog.py   Stage 2: RuvC-fold superfamily triage
│   ├── 41_stage1_pfam_filter.py          Stage 1 Pfam pre-filter
│   ├── 50_predictor_smoke_test.py        10-probe end-to-end smoke test
│   │
│   └── figures/
│       ├── fig1_taxonomy.py        Fig 1 — Tier-A / Tier-B taxonomy diagram
│       ├── fig2_confusion_matrix.py Fig 2 — Tier-A confusion matrix
│       ├── fig3_channel_ablation.py Fig 3 — Channel ablation bar chart
│       ├── fig4_is110_composite.py  Fig 4 — IS110 composite domain cartoon
│       ├── fig5_holdout_probes.py   Fig 5 — Holdout probe confidence scores
│       └── fig6_fanzor_catalog.py   Fig 6 — Fanzor/TnpB catalog distribution
│
├── tests/
│   ├── conftest.py                 Shared fixtures
│   ├── unit/
│   │   ├── test_aggregator.py      Evidence aggregator + IS110 override (22 tests)
│   │   ├── test_api_helpers.py     _build_feature_row, Prediction model (21 tests)
│   │   ├── test_composite_head.py  CompositeHead fit/predict/FP rate (6 tests)
│   │   ├── test_domain_features.py extract_domain_features, IS110 flags (14 tests)
│   │   ├── test_lightgbm_clf.py    LightGBMClassifier fit/predict/CI (7 tests)
│   │   └── test_seq_features.py    ESM-2 constants, embed_sequence (11 tests)
│   ├── integration/
│   │   ├── test_evidence_pipeline.py  Aggregate → label pipeline (4 tests)
│   │   └── test_predictor_api.py      Predictor.load → predict (29 tests; VM-gated for model probes; ISCro4 gate probe + gate_override test added)
│   └── regression/
│       ├── test_holdout_probes.py     5-probe OOD holdout (VM-gated; requires /data/models; ISCro4 gate path in test_predictor_api.py)
│       └── test_tier_a_macro_f1_drift.py  Tier-A macro-F1 ≥ 0.9862 baseline guard (VM-gated)
│
├── docs/
│   ├── conf.py                     Sphinx + Furo configuration
│   ├── index.rst                   Documentation root
│   ├── quickstart.rst              Installation and usage guide
│   ├── changelog.rst               Version history
│   └── api/
│       ├── predictor.rst           Predictor API reference
│       ├── features.rst            Feature module reference
│       └── models.rst              Model wrapper reference
│
├── containers/
│   └── structure/Dockerfile        pen-stack/structure Docker image (SaProt + Foldseek)
│
├── data/                           Curator review files (tracked in git)
│   ├── review_queue_annotated.tsv      28 manual review decisions
│   ├── review_queue_annotated_final.tsv Finalized curator decisions
│   └── review_queue_summary.json       Review queue statistics
│
├── results/                        Pre-computed summary JSONs
│   ├── holdout_results_corrected.json  Holdout validation final results
│   ├── fanzor_candidates_summary.json  Fanzor catalog summary statistics
│   └── is110_triage_summary.json       IS110 triage summary statistics
│
├── holdout_set.yaml                6 OOD probe definitions (sequences + expected labels; v0.5.3 adds ISCro4/D2TGM5)
├── pyproject.toml                  Package metadata, dependencies, pytest + coverage config
├── .readthedocs.yaml               ReadTheDocs build configuration
├── .github/workflows/
│   ├── ci.yml                      Unit + integration tests (Python 3.10, 3.11)
│   ├── docker.yml                  pen-stack/structure image build + push to GHCR
│   └── docs.yml                    ReadTheDocs trigger on main push
├── MODEL_CARD.md                   Model card: performance, limitations, intended use
├── LABEL_PROVENANCE.md             Gold-set label provenance and curation decisions
├── VALIDATION.md                   Pre-registered success criteria and results
├── UPDATE_STRATEGY.md              Model update and versioning policy
└── CITATION.cff                    Machine-readable citation metadata
```

---

## Trained models & data

Model artifacts (PKL files, feature matrices, catalogs) are distributed via Zenodo, not bundled in this repository. `Predictor.load()` downloads them automatically on first use.

**Zenodo deposit:** `https://zenodo.org/records/TODO_FILL_AFTER_DEPOSIT` (DOI pending)

| Artifact | Description | Size |
|---|---|---|
| `models/tier_a/model.pkl` | Tier-A 3-class LightGBM; macro-F1=0.9862 | 981 KB |
| `models/composite_head/model.pkl` | Binary IS110 composite head | 259 KB |
| `models/tier_b/DSB_NUCLEASE/model.pkl` | Tier-B sub-classifier | 350 KB |
| `models/tier_b/DSB_FREE_TRANSEST_RECOMBINASE/model.pkl` | Tier-B sub-classifier | 361 KB |
| `features/feature_matrix.parquet` | 1953-dim training matrix (572 proteins) | 6.6 MB |
| `features/esm2_150M_v6.parquet` | Pre-computed ESM-2 150M embeddings | 26 MB |
| `catalogs/fanzor_candidates.parquet` | 2,463 Fanzor/TnpB predictions | 370 KB |
| `catalogs/is110_triage.parquet` | 31,871 IS110-family protein predictions | 455 KB |

---

## Running the pipeline

All training and feature-extraction scripts require the VM environment (Docker, 64 GB RAM, GPU). Scripts are numbered and designed to be run sequentially.

```bash
# Evidence ingestion (scripts 01–08) — requires network access to M-CSA, Rhea, UniProt, etc.
python scripts/01_pull_mcsa.py
# ... through ...
python scripts/08_ingest_curator_decisions.py

# Feature computation (scripts 10–14) — requires Docker pen-stack/structure image
python scripts/10_compute_esm2_embeddings.py
python scripts/14_assemble_feature_matrix.py   # final 1953-dim assembly

# Training (scripts 20–24)
python scripts/20_train_tier_a.py              # seed=42, 5-fold stratified CV
python scripts/21_train_tier_b.py
python scripts/22_train_composite_head.py
python scripts/23_channel_ablation.py          # 7 ablation conditions
python scripts/24_bootstrap_cis.py             # 1000× bootstrap CIs

# Validation (script 30)
python scripts/30_holdout_validation.py        # 5-probe OOD holdout
```

## Testing

```bash
pip install -e ".[dev]"

# Unit + integration tests (no model files required)
pytest tests/unit/ tests/integration/ -v

# Full suite including regression (requires /data/models/ on VM)
pytest tests/ -v
```

**Test suite:** 103 unit + integration tests pass in CI (61% coverage). Regression tests are VM-gated and skip gracefully in CI.

---

## Citation

If you use mech-class, please cite both papers:

> Ahmed, A. (2026). *MECH-CLASS: Structure-aware mechanism classification for programmable genome-writing enzymes.* Briefings in Bioinformatics. (in submission)

> Ahmed, A. (2026). *GENOME-ATLAS: A unified knowledge graph for programmable genome-writing enzymes.* Nucleic Acids Research. (in submission)

## License

MIT — see [LICENSE](LICENSE).
