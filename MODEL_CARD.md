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
- **Holdout probes (OOD):** IS110 (A0A7C9VKZ0), Fanzor (Q8I6T1), SpCas9 (Q99ZW2), Bxb1 (Q9B086), Tn5 (Q46731)
- **In-distribution probe (composite FP check):** Cre (P06956) — in training set row 8658; reported for completeness

## Performance

| Metric | Value | 95% CI |
|---|---|---|
| Tier-A macro-F1 | See cv_results.json | Computed with 1000× bootstrap, seed=42 |
| IS110 tier-A confidence | ≥ 0.60 (required) | — |
| Fanzor tier-A confidence | ≥ 0.70 (required) | — |
| Composite FP rate | 25% (1/4; Cas9 FP, see Limitation 3) | OOD probes: Cas9/Bxb1/Cre/Tn5 |

*Holdout numbers: 5/5 Tier-A probes PASS. Composite FP rate FAILS pre-registered ≤10% gate.*

## Limitations

1. **Small training set (~150–200 proteins).** LightGBM is appropriate for this regime, but
   confidence scores should be interpreted cautiously for proteins far from any training example.
   Use the `novelty_score` output to flag high-novelty predictions.

2. **IS110 composite case is rule-assisted.** The composite head uses a domain presence rule
   (PF01548 + PF02371) as a strong prior. This makes the IS110 correction reliable but means
   proteins with atypical domain architectures may be missed.

3. **Composite head FP rate exceeds pre-registered threshold.** Pre-registered composite FP
   threshold (≤ 10% on Cas9/Bxb1/Cre/Tn5) was not met. Observed FP rate: 25% (1/4). The
   composite head learns a heuristic ('multiple Pfam domains → composite') from 14 multi-domain
   training positives. Cas9 (5 domains) triggers this heuristic at P=0.753. Users should treat
   the flag as a triage signal for IS110-like architecture, not a definitive non-composite
   classifier.

3. **Structure dependency.** F_struct and F_active_site channels require AlphaFold structures.
   Proteins without an AlphaFold model (non-reviewed UniProt, very short sequences, novel organisms)
   receive zero-filled structure channels. Tier-A accuracy is degraded ~5–15% (see channel ablation).

4. **Tier-B is supplementary.** Tier-B sub-classifiers are trained on very small per-class sets
   (5–30 proteins). Tier-B predictions should be treated as hypotheses, not ground truth.

5. **No wet-lab validation.** All predictions are computational. Experimental confirmation is
   required before using predictions for therapeutic or engineering applications.

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
