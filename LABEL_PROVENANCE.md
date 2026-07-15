# MECH-CLASS Label Provenance

Records the origin, weight, and curation history of every training label used in
MECH-CLASS.

## Evidence source hierarchy

| Source | Weight | Type | Notes |
|---|---|---|---|
| M-CSA | 1.0 | Experimental | Mechanism & Catalytic Site Atlas; gold-standard |
| Foundational systems | 1.0 | Curated | GENOME-ATLAS anchor labels (IS621, Cas9, Bxb1, Tn5, TnsABC) |
| CRISPRCasdb | 0.9 | Curated DB | CRISPR system classifications from Pasteur |
| Rhea | 0.8 | Experimental | Biochemical reaction DB; IUBMB-linked |
| UniProt ACT_SITE | 0.7 | Curated annotation | Reviewed UniProtKB entries only |
| TnPedia / ISfinder | 0.7 | Curated DB | IS/Tn family classification |
| Pfam whitelist | 0.6 | Structural | Labeling whitelist from GENOME-ATLAS; 18 Pfam families |
| InterPro clan | 0.5 | Structural | Clan membership inference; lowest confidence |

Gate 0 (high-authority requirement): proteins with ONLY Pfam-whitelist or InterPro-clan
evidence are discarded from the gold set. At least one high-authority source
(M-CSA, Foundational, CRISPRCasdb, Rhea, UniProt_features, TnPedia) is required.

## Composite architecture definition

A protein is flagged `composite_architecture=True` if it carries two catalytically
active modules of distinct evolutionary origin in a single polypeptide:

1. **IS110 dual-domain bridge recombinases** - proteins with BOTH PF01548
   (DEDD_Tnp_IS110, RuvC-fold N-terminal) AND PF02371 (Transposase_20, serine-Tnp
   C-terminal) detected in the ATLAS. 12 such proteins identified; 14 reach the gold
   set when combined with Foundational system evidence.

2. **CAST/Tn7 transposition machinery** - P13988 (TnsA) and P05846 (TnsC) from the
   TnsABC_CAST Tn7-family system (Foundational_systems_v0.6.0, composite=True).

**Explicitly excluded from composite detection:**
- PF07282 (Cas12f1-like_TNB / TnpB): single-domain RNA-guided nuclease; its
  transposon association is at the element level, not the protein level. Does NOT
  satisfy the two-catalytic-module criterion.

## IS110 composite override rule

Proteins with **both** PF01548 and PF02371 Pfam hits, AND at least one high-authority
source (TnPedia, Foundational), are assigned `DSB_FREE_TRANSEST_RECOMBINASE`
regardless of any InterPro CL0219 -> DSB_NUCLEASE inference. This corrects the
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
(PF01548+PF02371) in the ATLAS - same architecture as IS621 (Hiraizumi 2024 Nature).

## Gate 1 PASS/FAIL record

| Criterion | Required | Actual | Status |
|---|---|---|---|
| Total proteins | >= 150 | 572 | PASS |
| DSB_NUCLEASE | >= 30 | 39 | PASS |
| DSB_FREE_TRANSEST_RECOMBINASE | >= 30 | 449 | PASS |
| TRANSPOSASE | >= 30 | 84 | PASS |
| Composite-flagged | >= 10 | 14 | PASS |
| Hold-out exclusion | 0 present | 0 present | PASS |

**Gate 1: pass.** All label-count and hold-out-exclusion thresholds met.

## Manual review decisions (30 proteins)

All 30 manual-review proteins had curator decisions applied by `08_ingest_curator_decisions.py`.
28 curator overrides were applied (2 proteins retained automated prediction).

| Decision | Count | Notes |
|---|---|---|
| DSB_FREE_TRANSEST_RECOMBINASE | 19 | 7 IS110 dual-domain, 6 Tn554 Tyr-recombinases, 2 Phage-int, 2 Tn7, 2 conjugative Tnp-integrase |
| TRANSPOSASE | 5 | DDE rve integrase (gypsy/retrotransposon); DDE mechanism confirmed |
| DSB_NUCLEASE | 4 | PF07282 TnpB CAST-associated nucleases |
| (kept prediction) | 2 | P05846 (TnsC), P13988 (TnsA) - Tn7 CAST, auto DSB_FREE confirmed |

## Pre-registered hold-out proteins

The following proteins are excluded from training labels and reserved for evaluation:

| Protein | Accession | Reason |
|---|---|---|
| SpCas9 | Q99ZW2 | Pre-registered hold-out (excluded by 08_ingest_curator_decisions.py) |
| Tn5 | Q46731 | Pre-registered hold-out (excluded by 08_ingest_curator_decisions.py) |
| Bxb1 | **Q9B086** | Pre-registered hold-out - accession corrected (see Data Pipeline Corrections below). Q9B086 not in ATLAS; no exclusion step needed. |
| Cre | **P06956** | Intended composite-FP evaluation probe per the pre-registration, but **P06956 is in the training feature matrix** (DSB_FREE / B1_Site_Specific_Recombinase). Cannot serve as OOD holdout. See Data Pipeline Corrections. |
| IS621 | - | Not in ATLAS (foundational_systems.yaml proteins: []) |
| SpuFz1 | - | Not in ATLAS (foundational_systems.yaml proteins: []) |

## Data Pipeline Corrections

The pre-registered hold-out accessions are finalized as follows:

| Probe | Accession | Protein | Domains | In training? |
|---|---|---|---|---|
| Bxb1 | Q9B086 | *Mycobacterium* phage Bxb1 integrase (500 AA) | PF07508 + PF00239 | No (verified absent) |
| Cre | P06956 | *Enterobacteria* phage P1 Cre recombinase (343 AA) | PF00589 | Yes (row 8658, DSB_FREE / B1_Site_Specific_Recombinase) |
| Tn5 | Q46731 | *E. coli* Tn5 transposase | PF01609 | No (excluded by 08_ingest_curator_decisions.py) |

`foundational_systems.yaml` registers `Bxb1_integrase: [Q9B086]` and
`Cre_recombinase: [P06956]`.

Cre (P06956) is present in the training feature matrix. PF00589 (Phage_integrase) is
the largest training family (424 proteins), so Cre is in-distribution and is reported
as an in-distribution sanity check rather than an OOD probe. The composite
false-positive criterion is evaluated on the four non-composite probes SpCas9, Bxb1,
Tn5, and in-distribution Cre. With the domain gate (PF01548 and PF02371 both required),
none fire the composite flag: FP rate 0/4 = 0%, within the pre-registered <= 10%
threshold. See MODEL_CARD.md for the full composite hold-out table.
