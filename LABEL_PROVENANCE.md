# MECH-CLASS Label Provenance

Records the origin, weight, and curation history of every training label used in
MECH-CLASS v1.0.

## Evidence source hierarchy

| Source | Weight | Type | Notes |
|---|---|---|---|
| M-CSA | 1.0 | Experimental | Mechanism & Catalytic Site Atlas; gold-standard |
| Foundational systems | 1.0 | Curated | Paper 1 anchor labels (IS621, Cas9, Bxb1, Tn5, TnsABC) |
| CRISPRCasdb | 0.9 | Curated DB | CRISPR system classifications from Pasteur |
| Rhea | 0.8 | Experimental | Biochemical reaction DB; IUBMB-linked |
| UniProt ACT_SITE | 0.7 | Curated annotation | Reviewed UniProtKB entries only |
| TnPedia / ISfinder | 0.7 | Curated DB | IS/Tn family classification |
| Pfam whitelist v1.2.0 | 0.6 | Structural | Whitelist from Paper 1; 35 Pfam families |
| InterPro clan | 0.5 | Structural | Clan membership inference; lowest confidence |

Gate 0 (high-authority requirement): proteins with ONLY Pfam-whitelist or InterPro-clan
evidence are discarded from the gold set. At least one high-authority source
(M-CSA, Foundational, CRISPRCasdb, Rhea, UniProt_features, TnPedia) is required.

## Composite architecture definition

A protein is flagged `composite_architecture=True` if it carries two catalytically
active modules of distinct evolutionary origin in a single polypeptide:

1. **IS110 dual-domain bridge recombinases** — proteins with BOTH PF01548
   (DEDD_Tnp_IS110, RuvC-fold N-terminal) AND PF02371 (Transposase_20, serine-Tnp
   C-terminal) detected in the ATLAS. 12 such proteins identified; 14 reach the gold
   set when combined with Foundational system evidence.

2. **CAST/Tn7 transposition machinery** — P13988 (TnsA) and P05846 (TnsC) from the
   TnsABC_CAST Tn7-family system (Foundational_systems_v0.6.0, composite=True).

**Explicitly excluded from composite detection:**
- PF07282 (Cas12f1-like_TNB / TnpB): single-domain RNA-guided nuclease; its
  transposon association is at the element level, not the protein level. Does NOT
  satisfy the two-catalytic-module criterion.

## IS110 composite override rule

Proteins with **both** PF01548 and PF02371 Pfam hits, AND at least one high-authority
source (TnPedia, Foundational), are assigned `DSB_FREE_TRANSEST_RECOMBINASE`
regardless of any InterPro CL0219 → DSB_NUCLEASE inference. This corrects the
misassignment caused by the RNase H-like clan membership of the IS110 N-terminal domain.

## Reviewer action thresholds

| Action | Condition |
|---|---|
| `auto_accept` | Confidence >= 0.75 AND no contradiction AND has high-authority source |
| `manual_review` | 0.50 <= confidence < 0.75 OR contradiction flag AND has high-authority source |
| `discard` | No high-authority source OR confidence < 0.50 |

## Gold set composition (mechanism_labels_final.parquet)

**Run date:** 2026-05-04  
**Total proteins:** 572  
**Hold-outs excluded:** 2 (Q99ZW2 SpCas9, Q46731 Tn5)

### Tier A distribution

| Tier A | Count | % |
|---|---|---|
| DSB_FREE_TRANSEST_RECOMBINASE | 449 | 78.5% |
| TRANSPOSASE | 84 | 14.7% |
| DSB_NUCLEASE | 39 | 6.8% |

### Composite proteins (n=14)

| UniProt | Label | Source |
|---|---|---|
| P05846 | DSB_FREE | TnsABC_CAST (Foundational) |
| P13988 | DSB_FREE | TnsABC_CAST (Foundational) |
| P14322 | DSB_FREE | IS621_bridge_recombinase (Foundational) |
| P14707 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |
| P19257 | DSB_FREE | IS621_bridge_recombinase (Foundational) |
| P19780 | DSB_FREE | IS621_bridge_recombinase (Foundational) |
| P19834 | DSB_FREE | IS621_bridge_recombinase (Foundational) |
| P20665 | DSB_FREE | IS621_bridge_recombinase (Foundational) |
| P55615 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |
| P55626 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |
| P55643 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |
| Q45968 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |
| Q56897 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |
| Q9HI37 | DSB_FREE | IS110 dual-domain (TnPedia + Pfam) |

Note: P14322, P19257, P19780, P19834, P20665 were added to IS621_bridge_recombinase
in foundational_systems.yaml v0.6.0 based on confirmed dual-domain Pfam co-occurrence
(PF01548+PF02371) in the ATLAS — same architecture as IS621 (Hiraizumi 2024 Nature).

## Gate 1 PASS/FAIL record

| Criterion | Required | Actual | Status |
|---|---|---|---|
| Total proteins | >= 150 | 572 | PASS |
| DSB_NUCLEASE | >= 30 | 39 | PASS |
| DSB_FREE_TRANSEST_RECOMBINASE | >= 30 | 449 | PASS |
| TRANSPOSASE | >= 30 | 84 | PASS |
| Composite-flagged | >= 10 | 14 | PASS |
| Hold-out exclusion | 0 present | 0 present | PASS |

**GATE 1: PASS** — Part C (SaProt/ESM-2 feature extraction) approved.

## Manual review decisions (30 proteins)

All 30 manual-review proteins had curator decisions applied by `08_ingest_curator_decisions.py`.
28 curator overrides were applied (2 proteins retained automated prediction).

| Decision | Count | Notes |
|---|---|---|
| DSB_FREE_TRANSEST_RECOMBINASE | 19 | 7 IS110 dual-domain, 6 Tn554 Tyr-recombinases, 2 Phage-int, 2 Tn7, 2 conjugative Tnp-integrase |
| TRANSPOSASE | 5 | DDE rve integrase (gypsy/retrotransposon); DDE mechanism confirmed |
| DSB_NUCLEASE | 4 | PF07282 TnpB CAST-associated nucleases |
| (kept prediction) | 2 | P05846 (TnsC), P13988 (TnsA) — Tn7 CAST, auto DSB_FREE confirmed |

## Pre-registered hold-out proteins

The following proteins are excluded from training labels and reserved for evaluation:

| Protein | Accession | Reason |
|---|---|---|
| SpCas9 | Q99ZW2 | Pre-registered hold-out (excluded by 08_ingest_curator_decisions.py) |
| Tn5 | Q46731 | Pre-registered hold-out (excluded by 08_ingest_curator_decisions.py) |
| Bxb1 | **Q9B086** | Pre-registered hold-out — accession corrected (see §Data Pipeline Corrections below). Q9B086 not in ATLAS; no exclusion step needed. |
| Cre | **P06956** | Intended composite-FP evaluation probe per §0.5, but **P06956 is in the training feature matrix** (DSB_FREE / B1_Site_Specific_Recombinase). Cannot serve as OOD holdout. See §Data Pipeline Corrections. |
| IS621 | — | Not in ATLAS (foundational_systems.yaml proteins: []) |
| SpuFz1 | — | Not in ATLAS (foundational_systems.yaml proteins: []) |

## Data Pipeline Corrections

The following accession errors were discovered during hold-out evaluation (2026-05-06) and corrected prior to submission.

### Bxb1 integrase accession chain error

`foundational_systems.yaml` had `Bxb1_integrase: proteins: []` — no accession was ever
registered for this system. Two downstream scripts each supplied a hand-written accession
without UniProt validation:

| Script | Accession used | Actual protein | Error |
|---|---|---|---|
| `scripts/14_assemble_feature_matrix.py` HOLDOUT_SET | O25753 | *Helicobacter pylori* HP_1128 (84 AA, uncharacterised) | Wrong organism, wrong function, no Pfam |
| `scripts/figures/fig5_holdout_probes.py` | Q8VVR2 | *Staphylococcus aureus* GajA nuclease (380 AA, PF13175) | Wrong organism, nuclease not recombinase |

**Correction:** `Q9B086` (*Mycobacterium* phage Bxb1 integrase, 500 AA, PF07508+PF00239) is
the canonical Bxb1 hold-out probe. It is absent from the training feature matrix (verified).
The pre-registration intent was to test a large serine integrase; the accession was a
placeholder. This correction is documented as an accession fix, not a post-hoc probe change.
`foundational_systems.yaml` updated to `Bxb1_integrase: proteins: [Q9B086]`.

### Cre recombinase — proteins: [] corrected; P06956 found in training

`foundational_systems.yaml` had `Cre_recombinase: proteins: []`. The canonical probe is
`P06956` (*Enterobacteria* phage P1 Cre recombinase, 343 AA, PF00589 Phage_integrase).
`foundational_systems.yaml` updated to `Cre_recombinase: proteins: [P06956]`.

**However:** P06956 is present in the training feature matrix (row 8658, labeled
DSB_FREE_TRANSEST_RECOMBINASE / B1_Site_Specific_Recombinase). The PF00589
(Phage_integrase) family is the single largest class in the training set (424 proteins),
so Cre was included during standard label ingestion. P06956 was never excluded from training
by `08_ingest_curator_decisions.py` (only Q99ZW2 and Q46731 are explicitly excluded there;
O25753 was absent from ATLAS entirely).

**Consequence:** Cre cannot serve as an OOD composite evaluation probe. The pre-registered
composite FP criterion (FP rate ≤ 10% on Cas9, Bxb1, Cre, Tn5 holdouts) can only be
evaluated on 3 OOD probes: SpCas9 (FP), Bxb1/Q9B086 (TN), Tn5 (TN). FP rate = 1/3 = 33%.
The criterion fails. See MODEL_CARD.md §Composite head hold-out evaluation for full discussion.

## Version history

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-05 | Initial label set; n_sources counted rows not databases (9,500 labels — WRONG) |
| v0.2 | 2026-05 | Added HIGH_AUTHORITY_SOURCES gate; n_sources = unique DB families (569 labels) |
| v0.3 | 2026-05 | IS110-only composite (PF07282 reverted); hold-out exclusion added; IS110 proteins added to Foundational (572 labels, Gate 1 PASS) |
| v0.4 | 2026-05-06 | Accession corrections: Bxb1 O25753→Q9B086; Cre proteins:[]→[P06956]. foundational_systems.yaml updated. Hold-out table corrected (5/5 PASS Tier-A). P06956 found in training — Cre cannot be used as OOD composite probe. Composite FP rate = 1/3 = 33% on 3 OOD probes (fails ≤ 10% gate). |
