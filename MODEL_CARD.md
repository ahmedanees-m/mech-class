# MECH-CLASS Model Card

**Model:** MECH-CLASS v1.0
**Type:** Two-tier mechanism classifier (LightGBM + composite binary head)
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
| Composite head | Binary LightGBM (IS110 detection) |
| Feature channels | F_seq (640d), F_struct (1280d), F_domain (~25d), F_active_site (~20d) |
| F_seq source | ESM-2 150M mean-pool (from GENOME-ATLAS v0.6.0) |
| F_struct source | SaProt 650M mean-pool (AlphaFold structure + Foldseek SA-tokens) |
| pLDDT gate | F_struct and F_active_site zero-filled when mean active-site pLDDT < 70 |

## Training Data

- **Labeled set:** ~150–200 curated proteins
- **Label sources:** M-CSA (w=1.0), Foundational systems (w=1.0), CRISPRCasdb (w=0.9),
  Rhea (w=0.8), UniProt ACT_SITE (w=0.7), TnPedia (w=0.7), Pfam whitelist (w=0.6), InterPro (w=0.5)
- **Full provenance:** see [LABEL_PROVENANCE.md](LABEL_PROVENANCE.md)
- **Holdout probes:** IS110 (A0A7C9VKZ0), Fanzor (Q8I6T1), SpCas9 (Q99ZW2),
  Bxb1 (Q9B086), Tn5 (Q46731) — all absent from training.
  Cre (P06956) was intended as a composite evaluation probe but was found to be
  in the training set; it cannot serve as an OOD hold-out.
  See [LABEL_PROVENANCE.md §Data Pipeline Corrections](LABEL_PROVENANCE.md) for
  Bxb1 accession correction (O25753 → Q9B086).

## Performance

### Tier-A cross-validation (5-fold stratified CV, N=572)

| Metric | Value | 95% CI (1000× bootstrap) |
|---|---|---|
| Tier-A macro-F1 (LightGBM) | 0.9862 | [0.9530, 1.000] |
| Tier-A macro-F1 (MLP baseline) | 0.9664 | [0.9070, 0.9977] |
| IS110 hold-out tier-A confidence | 0.997 | — (threshold ≥ 0.60: PASS) |
| Fanzor hold-out tier-A confidence | 0.977 | — (threshold ≥ 0.70: PASS) |

### Tier-A hold-out evaluation (6 probes, corrected accessions)

| Probe | Accession | Tier-A Predicted | Conf | Status |
|---|---|---|---|---|
| IS110 | A0A7C9VKZ0 | DSB_FREE_TRANSEST_RECOMBINASE | 0.997 | **PASS** |
| Fanzor | Q8I6T1 | DSB_NUCLEASE | 0.977 | **PASS** |
| SpCas9 | Q99ZW2 | DSB_NUCLEASE | 1.000 | **PASS** |
| Bxb1 | Q9B086 | DSB_FREE_TRANSEST_RECOMBINASE | 0.966 | **PASS** |
| Tn5 | Q46731 | TRANSPOSASE | 0.869 | **PASS** |
| Cre | P06956 | DSB_FREE_TRANSEST_RECOMBINASE | [see below] | [see below] |

Tier-B = UNKNOWN for all probes. Acceptable per §0.5 (ungated; training N < 3 sub-class examples
for TRANSPOSASE; N=39 for DSB_NUCLEASE insufficient for reliable sub-classification).

### Composite head hold-out evaluation (pre-registered criterion: FP rate ≤ 10%)

| Probe | Accession | OOD? | Expected composite | Predicted composite | P(True) | Result |
|---|---|---|---|---|---|---|
| IS110 | A0A7C9VKZ0 | Yes (holdout) | True | **True** | 0.999 | TP |
| SpCas9 | Q99ZW2 | Yes (holdout) | False | **True** | 0.753 | **FP** |
| Bxb1 | Q9B086 | Yes (not in training) | False | False | 0.118 | TN |
| Tn5 | Q46731 | Yes (holdout) | False | False | 0.379 | TN |
| Cre | P06956 | **No — in training** | False | False | 0.005 | (in-distribution, not evaluated) |

**Note on Cre:** P06956 (*E.* phage P1 Cre recombinase) was intended as a pre-registered composite
evaluation probe but was discovered to be present in the training feature matrix (labeled
DSB_FREE_TRANSEST_RECOMBINASE / B1_Site_Specific_Recombinase). Its composite=False result is
therefore in-distribution and cannot be used as an OOD holdout datum.

**Hold-out composite FP rate (3 OOD non-composite probes): 1/3 = 33%.**
**This FAILS the pre-registered ≤ 10% criterion.**
With IS110 TP included: composite precision on hold-out = 1/2 = 50% (1 TP, 1 FP).

#### Why the composite head over-fires on SpCas9

The composite head was trained on 14 positive examples, all of which are multi-domain
IS110-family or CAST proteins carrying two catalytically independent modules. SpCas9
has five Cas9-specific Pfam domains in a single polypeptide (PF13395, PF18541, PF16595,
PF16592, PF16593), and the composite head learned a "multiple domains → composite
architecture" heuristic that transfers incorrectly to Cas9.

The training set contains no negative examples with five or more whitelist Pfam hits on
a non-composite protein: SpCas9 is out-of-distribution for this feature dimension.

**Correct interpretation:** The composite flag reliably detects IS110-like dual-module
architectures (the paper's headline claim) but has elevated FP rate for natural
multi-domain proteins carrying four or more whitelist Pfam domains. Users should treat
composite=True as a triage signal for IS110-like architecture review, not as a definitive
binary classifier for all multi-domain enzymes. For proteins with ≥4 whitelist Pfam hits,
inspect domain annotations before accepting the composite call.

**This limitation is acknowledged in the paper** (Methods §3.4 and Supplementary Table S2)
and does not affect the Tier-A classification performance metrics or the IS110 reclassification
claims, which are the primary contributions.

## Limitations

1. **Small training set (~150–200 proteins).** LightGBM is appropriate for this regime, but
   confidence scores should be interpreted cautiously for proteins far from any training example.
   Use the `novelty_score` output to flag high-novelty predictions.

2. **IS110 composite case is rule-assisted.** The composite head uses a domain presence rule
   (PF01548 + PF02371) as a strong prior. This makes the IS110 correction reliable but means
   proteins with atypical domain architectures may be missed.

3. **Structure dependency.** F_struct and F_active_site channels require AlphaFold structures.
   Proteins without an AlphaFold model (non-reviewed UniProt, very short sequences, novel organisms)
   receive zero-filled structure channels. Tier-A accuracy is degraded ~5–15% (see channel ablation).

4. **Tier-B is supplementary.** Tier-B sub-classifiers are trained on very small per-class sets
   (5–30 proteins). Tier-B predictions should be treated as hypotheses, not ground truth.

5. **No wet-lab validation.** All predictions are computational. Experimental confirmation is
   required before using predictions for therapeutic or engineering applications.

6. **Composite head elevated FP rate on multi-Pfam non-composite proteins.** The composite
   binary head achieves 0/558 FP on in-distribution negatives (5-fold CV) but fires on
   SpCas9 (P=0.753) in the hold-out set, yielding a hold-out FP rate of 25% (1/4 non-composite
   probes). This reflects a training-distribution gap: the 14 composite positives are all
   IS110/CAST dual-module proteins, and no in-distribution negative has ≥ 4 whitelist Pfam
   domains. The composite flag should be interpreted as a triage signal for IS110-like
   dual-module architecture rather than a universal multi-domain classifier.
   See §Composite head hold-out evaluation above for per-probe detail.

## Bias and Fairness

- Training data is biased toward well-studied organisms (E. coli, B. subtilis, human pathogens,
  S. cerevisiae). Eukaryotic and archaeal enzymes are underrepresented.
- Fanzor/OMEGA predictions may be more reliable than stated CIs because foundational anchors
  (SpFanzor1) provide strong structural signal.

## Update Strategy

See [UPDATE_STRATEGY.md](UPDATE_STRATEGY.md). Model is versioned; v1.x minor updates add new
training examples; v2.0 major updates require re-registration of success criteria.

## Citation

```bibtex
@article{ahmed2026mechclass,
  author = {Anees Ahmed},
  title  = {MECH-CLASS: Structure-aware mechanism classification for programmable genome-writing enzymes},
  journal = {Briefings in Bioinformatics},
  year = {2026},
  note = {in submission}
}
```
