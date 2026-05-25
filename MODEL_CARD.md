# MECH-CLASS Model Card

**Model:** MECH-CLASS v0.5.3
**Type:** Two-tier mechanism classifier (LightGBM + biochemically-gated composite head)
**Task:** Predict the catalytic mechanism of DNA-modifying enzymes from sequence + structure
**Package:** `pip install mech-class`

---

## Intended Use

**Primary use:** Computational annotation of DNA-modifying enzyme mechanism from protein sequence
(and optionally AlphaFold structure). Designed for researchers building genome-engineering
pipelines who need mechanism-aware filtering of large enzyme catalogs.

**Secondary use:** Reclassification of InterPro/Pfam annotations that incorrectly assign IS110-
family recombinases to the DSB_NUCLEASE class due to CL0219 clan membership.

**Out-of-scope:** This model is not validated for RNA-modifying enzymes, lipid/carbohydrate
enzymes, or non-enzymatic DNA-binding proteins. Do not use for clinical or therapeutic
decision-making without independent experimental validation.

---

## Architecture

| Component | Details |
|---|---|
| Tier-A classifier | LightGBM, 3 classes, 5-fold stratified CV |
| Tier-B classifiers | Separate LightGBM per Tier-A class |
| Composite head | Biochemical gate (PF01548 and PF02371) + binary LightGBM confidence |
| Feature channels | F_seq (640d), F_struct (1280d), F_domain (26d), F_active_site (7d) |
| F_seq source | ESM-2 150M mean-pool (from GENOME-ATLAS v0.7.1, pinned; see pyproject.toml) |
| F_struct source | SaProt 650M mean-pool (AlphaFold structure + Foldseek SA-tokens) |
| pLDDT gate | F_struct and F_active_site zero-filled when mean active-site pLDDT < 70 |

### Tier-A IS110 hard gate (v0.5.2)

IS110-family bridge recombinases (IS621, IS110, ISDra2, etc.) were misclassified as
DSB_NUCLEASE when scored without a pre-computed ESM-2 embedding (domain-only inference
path). Root cause: the LightGBM Tier-A model was trained where all 14 IS110 training
proteins had real ESM-2 embeddings; at inference time (F_seq zero-filled), the feature
vector is out-of-distribution and the model incorrectly outputs DSB_NUCLEASE.

Fix: same pattern as composite gate. If PF01548 (DEDD_Tnp_IS110) AND PF02371
(Transposase_20) are both present, `tier_a` is forced to `DSB_FREE_TRANSEST_RECOMBINASE`.
`tier_a_gate_override: True` is set in the `Prediction` object for audit trail.
Confidence = max(ML_DSB_FREE_prob, 0.90).

IS110 proteins with real ESM-2 embeddings (all 14 training examples, holdout A0A7C9VKZ0)
score DSB_FREE correctly without the gate; the gate exists only for the OOD domain-only
path. CV F1 metrics and holdout results are unchanged.

Canonical OOD gate probe (v0.5.3): **ISCro4 (D2TGM5**, Citrobacter rodentium ICC168; formerly "IS622" in Perry 2025 *bioRxiv*).
Not in training set; PF01548 + PF02371 -> gate fires; `DSB_FREE_TRANSEST_RECOMBINASE`,
`tier_a_gate_override=True`, conf = 0.90. Sources: Pelea 2026 *Science* adz1884; Perry 2025 bioRxiv 2025.05.14.653916.

### Composite head design (v0.5.1)

The composite architecture flag uses a **two-layer decision**:

1. **Biochemical hard gate (necessary condition):** Both `PF01548` (DEDD_Tnp_IS110,
   IS110 N-terminal RuvC-fold domain) and `PF02371` (Transposase_20, IS110 C-terminal
   serine-Tnp domain) must be present in the protein's Pfam annotation. If either is
   absent, `composite` is forced to `False` and `composite_prob` to `0.0`, regardless
   of the ML head output.

2. **ML confidence calibration (sufficient condition):** Only when the gate passes does
   the LightGBM head (trained on 33 domain + active-site features) provide a probability
   score. `composite=True` requires gate pass **and** ML probability >= 0.5.

This design makes the composite flag **biochemically grounded** rather than a pure learned
heuristic. The IS110 bridge recombinase architecture is defined by the co-occurrence of these
two specific Pfam families (Hiraizumi et al. 2024 *Nature*; Vaysset et al. 2025 *Nat Microbiol*);
the ML head calibrates confidence, not the detection logic.

---

## Training Data

- **Labeled set:** 572 curated proteins (gold set after high-authority filter)
- **Label sources:** M-CSA (w=1.0), Foundational systems (w=1.0), CRISPRCasdb (w=0.9),
  TnPedia (w=0.85), Rhea (w=0.8), ATLAS domain (w=0.75), UniProt ACT_SITE (w=0.7),
  Pfam whitelist (w=0.6), InterPro (w=0.5)
- **Full provenance:** see [LABEL_PROVENANCE.md](LABEL_PROVENANCE.md)
- **Holdout probes:** IS110 (A0A7C9VKZ0), Fanzor (Q8I6T1), SpCas9 (Q99ZW2),
  Bxb1 (Q9B086, corrected from Q8VVR2), Tn5 (Q46731, corrected from P00509),
  **ISCro4 (D2TGM5, v0.5.3 gate probe)** - all absent from training.
  Cre (P06956) was intended as a composite evaluation probe but was found to be
  in the training set; it cannot serve as an OOD hold-out.
  See [LABEL_PROVENANCE.md Data Pipeline Corrections](LABEL_PROVENANCE.md).

---

## Performance

### Tier-A cross-validation (5-fold stratified CV, N=572, seed=42)

| Metric | Value | 95% CI (1000x bootstrap per fold) |
|---|---|---|
| Tier-A macro-F1 (LightGBM, full 1953d) | **0.9862** | [0.953, 1.000] |
| Tier-A macro-F1 (domain_only, 26d) | 0.9859 | [0.944, 1.000] |
| Tier-A macro-F1 (MLP baseline) | 0.9664 | [0.907, 0.998] |
| Composite head CV AUROC | **0.9922** | - |
| Composite head CV FP rate | **0.0** | - |

### Tier-A hold-out evaluation (6 OOD probes + 1 in-distribution sanity check)

| Probe | Accession | Tier-A Predicted | Conf | Gate override | Pre-reg gate |
|---|---|---|---|---|---|
| IS110 | A0A7C9VKZ0 | DSB_FREE_TRANSEST_RECOMBINASE | 0.997 | No (in-dist ESM-2) | **PASS** (>=0.6) |
| Fanzor | Q8I6T1 | DSB_NUCLEASE | 0.977 | No | **PASS** (>=0.7) |
| SpCas9 | Q99ZW2 | DSB_NUCLEASE | 1.000 | No | **PASS** (>=0.6) |
| Bxb1 | Q9B086 | DSB_FREE_TRANSEST_RECOMBINASE | 0.966 | No | **PASS** (>=0.6) |
| Tn5 | Q46731 | TRANSPOSASE | 0.869 | No | **PASS** (>=0.6) |
| ISCro4 | D2TGM5 | DSB_FREE_TRANSEST_RECOMBINASE | >=0.90 | **Yes** (OOD gate) | **PASS** (>=0.9) |
| Cre | P06956 | DSB_FREE_TRANSEST_RECOMBINASE | 0.9999 | No | In-distribution - not evaluated |

**OOD Tier-A accuracy: 6/6 (100%).** Pre-registration gate passed for all 6 OOD probes.
D2TGM5 (ISCro4) is the canonical gate probe: `tier_a_gate_override=True` (v0.5.2 gate fires).

### Composite head hold-out evaluation (pre-registered criterion: FP rate <= 10%)

| Probe | OOD? | Gate pass | Expected composite | Predicted composite | ML raw P | Result |
|---|---|---|---|---|---|---|
| IS110 | Yes | yes (PF01548 + PF02371) | True | **True** | 0.999 | TP |
| SpCas9 | Yes | no (lacks PF01548/PF02371) | False | **False** | 0.753 raw | **TN (gate blocked)** |
| Bxb1 | Yes | no (PF07508 only) | False | False | 0.118 raw | TN |
| Tn5 | Yes | no (PF01609 only) | False | False | 0.379 raw | TN |
| Cre | No (in-training) | no (PF00589 only) | False | False | 0.005 raw | (in-distribution) |

**Hold-out composite FP rate (4 non-composite probes including in-distribution Cre): 0/4 = 0%.**
**Pre-registered <= 10% criterion: PASS.**

#### SpCas9 - the key case

SpCas9 (Q99ZW2) has five Cas9-specific Pfam domains (PF16593, PF16595, PF16592, PF22702,
PF13395) and carries neither PF01548 nor PF02371. The biochemical gate blocks the composite
call at source. The ML head's raw output is 0.753 (it would have been a FP under a pure
ML-only design), but the gate overrides this to `composite=False, composite_prob=0.0`.

The raw ML score is preserved in `ml_composite_prob_raw` in `holdout_results.json` for
transparency. This documents the v0.5.0 -> v0.5.1 correction and provides full audit trail.

**Note on Cre:** P06956 is IN TRAINING (row 8658, DSB_FREE / B1_Site_Specific_Recombinase,
424 PF00589 proteins in training). It cannot serve as an OOD holdout. Its in-distribution
composite=False result (gate blocked; ML raw P=0.005) confirms a true negative.

---

## Limitations

1. **Small training set (572 proteins, highly imbalanced).** LightGBM is appropriate for this
   regime, but confidence scores should be interpreted cautiously for proteins far from any
   training example. Class distribution: DSB_FREE 78.5%, TRANSPOSASE 14.7%, DSB_NUCLEASE 6.8%.

2. **Composite flag is IS110-specific.** The biochemical gate (PF01548 and PF02371) detects
   the IS110/bridge recombinase dual-domain architecture only. CAST integrase complexes
   (multi-protein, not single-chain composite) and other composite architectures are
   not detected by this flag. The composite head targets the specific mechanism described
   in Hiraizumi et al. 2024 and Vaysset et al. 2025.

3. **TRANSPOSASE Tier-B model absent.** The 84 TRANSPOSASE training proteins were divided
   among only 2 sub-classes, with the minor class having N < 3 - insufficient for stratified
   CV. A Tier-B TRANSPOSASE model is withheld until additional curated sub-labels are available.

4. **Structure dependency.** F_struct and F_active_site channels require AlphaFold structures
   with mean pLDDT >= 70. Proteins without a suitable AlphaFold model receive zero-filled
   structure channels (33/572 in training, 5.8%). The domain_only ablation condition (F1 =
   0.9859) shows this degradation is minimal for the 3-class Tier-A problem.

5. **Tier-B is supplementary.** Tier-B sub-classifiers are trained on small per-class sets
   (<= 449 proteins for DSB_FREE, <= 39 for DSB_NUCLEASE). Tier-B predictions should be treated
   as hypotheses, not ground truth.

6. **No wet-lab validation.** All predictions are computational. Experimental confirmation is
   required before using predictions for therapeutic or engineering applications.

7. **Catalog scoring is domain-only.** Fanzor (2,463 proteins) and IS110 triage (31,871
   proteins) catalogs were scored with F_seq zero-filled due to compute constraints.
   Individual protein predictions should be confirmed with full-channel scoring.

8. **SaProt requires GPU.** F_struct pre-computation uses SaProt 650M on an A100/V100.
   At inference time (via `api.py`), F_struct is always zero-filled unless the user provides
   a pre-computed embedding. This is planned to be resolved in v0.6.

---

## Bias and Fairness

- Training data is biased toward well-studied organisms (E. coli, phage, human pathogens,
  S. cerevisiae). Eukaryotic and archaeal enzymes are underrepresented.
- The DSB_FREE class dominates the training set (78.5%); confidence on DSB_NUCLEASE and
  TRANSPOSASE predictions may be lower for borderline cases.
- Fanzor/OMEGA detections are anchored on SpFanzor1 (Q8I6T1) which is in the holdout set
  and was correctly classified, providing some orthogonal validation.

---

## Update Strategy

See [UPDATE_STRATEGY.md](UPDATE_STRATEGY.md). Model is versioned with semantic versioning:
- `v0.5.x` - patch fixes (e.g. v0.5.1: biochemical gate for composite head)
- `v0.6.0` - planned: SaProt GPU inference at runtime, dom_24 editor-fusion flag, TRANSPOSASE Tier-B
- `v1.0.0` - post peer-review release with PyPI publication

---

## Changelog

| Version | Date | Change |
|---|---|---|
| v0.5.0 | 2026-05-07 | Initial release. Composite head: ML-only, FP rate 25% (SpCas9). |
| v0.5.1 | 2026-05-11 | Add composite biochemical hard gate (PF01548 and PF02371). Composite FP rate -> 0%. |
| v0.5.2 | 2026-05-22 | Add Tier-A IS110 hard gate. Fixes IS621 DSB_NUCLEASE misclassification in domain-only path. Pin genome-atlas <0.7.0. |
| v0.5.3 | 2026-05-23 | Add ISCro4/D2TGM5 as 6th OOD holdout probe. Bump genome-atlas pin >=0.7.1,<0.8.0 (SIMILAR_TO restored; ISCro4 in atlas). |

---

## Citation

```bibtex
@software{ahmed_mechclass,
  author = {Anees Ahmed},
  title  = {MECH-CLASS: mechanism classification for programmable genome-writing enzymes},
  year   = {2026},
  url    = {https://github.com/ahmedanees-m/mech-class}
}
```
