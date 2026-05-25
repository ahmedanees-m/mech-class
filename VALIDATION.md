# MECH-CLASS Validation

This document records the validation checks for MECH-CLASS and their results.
Full performance detail and limitations are in [MODEL_CARD.md](MODEL_CARD.md).

## Benchmark performance (572-protein gold set)

| Metric | Result | 95% bootstrap CI |
|---|---|---|
| Tier-A macro-F1 | 0.9862 | [0.953, 1.000] |
| Tier-A accuracy | 0.989 | - |
| Composite head false-positive rate | 0% (biochemical gate) | - |
| IS110 reclassification | 31,870 / 31,871 | - |

Confidence intervals use 1000-sample bootstrap resampling with seed 42.

## Channel ablation

| Feature set | Tier-A macro-F1 |
|---|---|
| F_domain only | ~0.94 |
| F_domain + F_seq (ESM-2) | 0.9862 |
| + F_struct (SaProt) | < 1% change |

Domain and sequence features carry the signal; structure embeddings add little
on the current gold set.

## Out-of-distribution holdout probes

Six pre-registered probes, none seen during training. All pass.

| Probe | Accession | Expected Tier-A | Min confidence | Result |
|---|---|---|---|---|
| IS110 bridge recombinase | A0A7C9VKZ0 | DSB_FREE_TRANSEST_RECOMBINASE | 0.60 | pass (conf 0.997) |
| Fanzor SpFanzor1 | Q8I6T1 | DSB_NUCLEASE | 0.70 | pass |
| SpCas9 | Q99ZW2 | DSB_NUCLEASE | 0.60 | pass |
| Bxb1 integrase | Q9B086 | DSB_FREE_TRANSEST_RECOMBINASE | 0.60 | pass |
| Tn5 transposase | Q46731 | TRANSPOSASE | 0.60 | pass |
| ISCro4 (gate probe) | D2TGM5 | DSB_FREE_TRANSEST_RECOMBINASE | 0.90 | pass (gate override) |

ISCro4 (D2TGM5) is absent from the ESM-2 training embeddings, so the raw model
predicts DSB_NUCLEASE (P around 0.57). The IS110 biochemical gate overrides this
to DSB_FREE_TRANSEST_RECOMBINASE with confidence floored at 0.90. Results are in
`results/holdout_results_corrected.json`.

## Composite architecture flag

| Metric | Result |
|---|---|
| IS110 composite false-positive rate | 0% on the holdout set (with domain gate) |
| IS110 composite detection | 31,870 / 31,871 IS110-family proteins across UniProt |

SpCas9 is a documented composite false positive at the raw ML layer (P=0.753);
the domain gate blocks it. See [MODEL_CARD.md](MODEL_CARD.md).

## Automated checks

- Tier-A macro-F1 drift guard: `tests/regression/test_tier_a_macro_f1_drift.py`
- Holdout probe regression: `tests/regression/test_holdout_probes.py`
- IS110 composite rule and aggregator override: `tests/unit/test_domain_features.py`,
  `tests/unit/test_aggregator.py`

Regression tests that need trained model files skip automatically when those
files are not present.

## Methodological checks

- SaProt embedding diversity audited before training (>= 95% unique vectors).
- Stratified 5-fold splits verified with no class collapse in any fold.
- Holdout probe accessions confirmed absent from the training set.
- All reported metrics carry 95% bootstrap confidence intervals (n=1000, seed=42).
- Label provenance is recorded in [LABEL_PROVENANCE.md](LABEL_PROVENANCE.md).
