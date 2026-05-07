Changelog
=========

v0.5.0 (2026-05-07)
--------------------

**First public pre-release.**

New features
^^^^^^^^^^^^
* ``Predictor.load()`` + ``predict_from_sequence()`` API for single-protein inference.
* ``Predictor.predict_from_fasta()`` for FASTA-file batch prediction.
* ``Predictor.predict_batch()`` for DataFrame batch prediction.
* ESM-2 150M singleton (lazy-loaded, CPU-only) for F_seq channel at inference time.
* Pfam domain features (26-dim; F_domain channel) with UniProt REST fallback.
* Composite architecture flag (IS110 binary head; ``composite_prob`` field).
* Tier-B sub-class labels (supplementary; not gated in pre-registration).

Model performance
^^^^^^^^^^^^^^^^^
* Tier-A OOD holdout: 5/5 probes PASS (IS110, Fanzor, SpCas9, Bxb1, Tn5).
* Tier-A 5-fold CV macro-F1: 0.9862 (pre-registered baseline).
* IS110 reclassification: 31,870/31,870 (99.9%) of PF01548+PF02371 proteins
  correctly assigned DSB_FREE_TRANSEST_RECOMBINASE.
* Composite FP rate: 25% (1/4 probes; SpCas9 FP documented, see MODEL_CARD.md).

Known limitations
^^^^^^^^^^^^^^^^^
* SaProt structure channel (F_struct) zero-filled at inference (GPU/PDB required).
* IS3-family TnpA bipartite architecture: embedded near IS110 in ESM-2 space;
  may be mis-classified as DSB_NUCLEASE. See MODEL_CARD.md Limitation 4.
* Zenodo model deposit URL pending peer review (``_download_from_zenodo()`` stub).
* Composite head FP rate (25%) exceeds pre-registered ≤10% threshold.

v0.4.0 (internal)
------------------
Training script completion (scripts 01–24). Label provenance, feature matrix,
model training, holdout validation.
