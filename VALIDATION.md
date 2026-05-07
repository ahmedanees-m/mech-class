# MECH-CLASS Validation

Documents all validation checks and their results for MECH-CLASS v1.0.

## Pre-training gates

| Check | Script | Status |
|---|---|---|
| SaProt embedding diversity audit (≥95% unique) | 11_compute_saprot_embeddings.py | RUN BEFORE TRAINING |
| ESM-2 coverage for all labeled proteins | 10_compute_esm2_embeddings.py | RUN BEFORE TRAINING |
| IS110 composite rule fires on known IS110 proteins | tests/unit/test_domain_features.py | AUTOMATED |
| Aggregator IS110 override logic | tests/unit/test_aggregator.py | AUTOMATED |

## Cross-validation (5-fold stratified)

Results written to `/data/models/tier_a/cv_results.json` after running `20_train_tier_a.py`.

| Metric | Pre-registered threshold | Result |
|---|---|---|
| Tier-A macro-F1 (point) | ≥ 0.80 | (fill after training) |
| 95% CI lower bound | ≥ 0.70 | (fill after training) |

## Holdout probe validation

Results written to `/data/validation/holdout_results.json` after running `30_holdout_validation.py`.

| Probe | Accession | Expected Tier-A | Min confidence | Result |
|---|---|---|---|---|
| IS110 representative | A0A7C9VKZ0 | DSB_FREE_TRANSEST_RECOMBINASE | 0.60 | (fill) |
| SpFanzor1 | Q8I6T1 | DSB_NUCLEASE (N2_Fanzor_OMEGA) | 0.70 | (fill) |
| SpCas9 | Q99ZW2 | DSB_NUCLEASE (N1_CRISPR_Cas) | 0.60 | (fill) |
| Bxb1 | Q8VVR2 | DSB_FREE_TRANSEST_RECOMBINASE (B3) | 0.60 | (fill) |
| Tn5 | P00509 | TRANSPOSASE (T1_DDE) | 0.60 | (fill) |

## Composite head

| Metric | Threshold | Result |
|---|---|---|
| IS110 composite FP rate | ≤ 10% | (fill after training) |
| IS110 composite recall (on holdout) | — | (fill) |

## Channel ablation

Results written to `/data/models/ablation/ablation_results.json` after running `23_channel_ablation.py`.

Required to appear in paper: F1 for F_seq, F_seq+F_struct, F_seq+F_struct+F_domain, full.

## Regression tests

Automated regression tests in `tests/regression/test_holdout_probes.py` run against trained model.
These are skipped on laptop (no model); run on VM after training.

## Honesty audit

Per Paper 1 §1.4.4 honesty disciplines carried forward:
- [ ] SaProt embedding collapse audit completed before training
- [ ] Stratified splits verified (no class collapses in any fold)
- [ ] Prospective set confirmed no overlap with training accessions
- [ ] All performance numbers report 95% bootstrap CIs (n=1000, seed=42)
- [ ] LABEL_PROVENANCE.md shipped with package
- [ ] MODEL_CARD.md shipped with package
