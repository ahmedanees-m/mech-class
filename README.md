# mech-class

[![CI](https://github.com/ahmedanees-m/mech-class/workflows/CI/badge.svg)](https://github.com/ahmedanees-m/mech-class/actions)
[![codecov](https://codecov.io/gh/ahmedanees-m/mech-class/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmedanees-m/mech-class)
[![PyPI](https://img.shields.io/pypi/v/mech-class.svg)](https://pypi.org/project/mech-class/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://mech-class.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Part of [PEN-STACK](https://github.com/ahmedanees-m)**

Mechanism classifier for DNA-modifying enzymes — predicts catalytic mechanism (three Tier-A classes + nine Tier-B sub-classes) from protein sequence + structure, with **explicit handling of composite catalytic architectures** such as IS110 family proteins (RuvC-fold N-terminal domain + serine-recombinase C-terminal domain).

Built on top of [GENOME-ATLAS](https://github.com/ahmedanees-m/genome-atlas) (Paper 1 of PEN-STACK).

## Scientific contribution

The single most novel contribution of MECH-CLASS is explicit handling of composite catalytic architectures. IS110 family proteins (bridge recombinases) carry a RuvC-fold DEDD N-terminal domain and a serine-recombinase Tnp C-terminal domain. Existing Pfam/BLAST classifiers either collapse this to "DEDD nuclease" (wrong) or "serine recombinase" (incomplete). MECH-CLASS predicts:

- Tier A: `DSB_FREE_TRANSEST_RECOMBINASE` ✓ (correct — no DSB, transesterase chemistry)
- Tier B: `B3_Programmable_Recombinase` ✓
- `composite_architecture: True` ✓ (RuvC-fold + serine Tnp, both domains required)

**Three Tier-A mechanism classes:**
- `DSB_NUCLEASE` — hydrolytic phosphodiester cleavage, DSB produced (Cas9, Cas12a/f, Fanzor)
- `DSB_FREE_TRANSEST_RECOMBINASE` — transesterification, no DSB (IS110, CAST, Cre, Bxb1)
- `TRANSPOSASE` — DDE-family cut-and-paste transposition (Tn5, IS3-family)

**Pre-registered success criteria** (locked 2026-04-30):
- Tier A macro-F1 ≥ 0.80, 95% bootstrap CI lower bound ≥ 0.70
- IS110 hold-out: `DSB_FREE_TRANSEST_RECOMBINASE` confidence ≥ 0.60
- Fanzor hold-out: `DSB_NUCLEASE` confidence ≥ 0.70
- Composite FP rate on Cas9/Bxb1/Cre/Tn5 ≤ 10%

## Install

```bash
pip install mech-class
```

## Quickstart

```python
from mech_class import Predictor

predictor = Predictor.load()
prediction = predictor.predict_from_sequence(
    accession="P0DOC6",
    sequence="MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNL...",
)
print(prediction.tier_a)              # 'DSB_NUCLEASE'
print(prediction.tier_b)              # 'N1_CRISPR_Cas'
print(prediction.composite)           # False
print(prediction.confidence)          # 0.94
```

Composite-case prediction (IS110 bridge recombinase):

```python
is110_pred = predictor.predict_from_sequence(
    accession="A0A0X1KFI0",  # IS110 family protein
    sequence="...",
)
print(is110_pred.tier_a)              # 'DSB_FREE_TRANSEST_RECOMBINASE'
print(is110_pred.tier_b)              # 'B3_Programmable_Recombinase'
print(is110_pred.composite)           # True
print(is110_pred.composite_evidence)  # ['RuvC-fold DEDD N-term', 'serine Tnp C-term']
```

## Command-line interface

```bash
# Predict mechanism for all sequences in a FASTA file
mech-class predict proteins.fasta --output predictions.parquet

# Apply to Fanzor ortholog catalog
mech-class predict fanzor_orthologs.fasta --output fanzor_predictions.parquet --device cuda
```

## Architecture

```
mech_class/
├── api.py              Predictor class (public entry point)
├── data/               Taxonomy YAML, data loaders
├── evidence/           Label construction: M-CSA, Rhea, UniProt, InterPro, TnPedia
├── features/           F_seq (ESM-2 reuse), F_struct (SaProt 650M), F_domain, F_active_site
├── models/             LightGBM Tier-A + Tier-B + composite flag head
└── utils/              pLDDT filter, active-site geometry helpers

scripts/
├── 01-07_*             Evidence ingestion + label aggregation + review queue
├── 10-13_*             Feature computation (SaProt, active-site, domain, fusion)
├── 20-24_*             Classifier training, ablation, bootstrap CIs
├── 30_holdout_*        Composite-case validation (IS110, Fanzor, Cas9, Bxb1, Tn5)
├── 40_fanzor_*         Prospective: ~3,000 Fanzor orthologs
└── 41_ruvc_fold_*      Prospective: ~800k RuvC-fold superfamily → top-1,000 catalog
```

## Feature channels

| Channel | Source | Dimension | Notes |
|---|---|---|---|
| F_seq | ESM-2 150M (Paper 1 reuse) | 640 | No re-inference |
| F_struct | SaProt 650M | 1280 | Structure-aware PLM |
| F_domain | Pfam binary + co-occurrence | ~40 | DEDD + IS110-specific composite flag |
| F_active_site | Geometry from PDB/AF (pLDDT ≥ 70) | ~20 | Active-site Ca distances, DSSP |

## Citation

If you use mech-class, please cite both papers:

> Ahmed, A. (2026). *MECH-CLASS: Structure-aware mechanism classification for programmable genome-writing enzymes.* Briefings in Bioinformatics. (in submission)

> Ahmed, A. (2026). *GENOME-ATLAS: A unified knowledge graph for programmable genome-writing enzymes.* Nucleic Acids Research. (in submission)

## License

MIT — see [LICENSE](LICENSE).
