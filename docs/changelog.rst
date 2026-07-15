Changelog
=========

v0.5.4 (2026-05-25)
-------------------

Changed
^^^^^^^
* Renamed the ISCro4 holdout probe to its canonical name (UniProt D2TGM5;
  Pelea et al. 2026 *Science* adz1884) across ``holdout_set.yaml``,
  ``holdout_results.json``, and ``test_predictor_api.py``. The deprecated
  preprint label is retained in ``aliases`` fields. No model behaviour change.
* Bumped the ``genome-atlas`` pin to ``>=0.7.2,<0.8.0`` for canonical ISCro4
  naming and the ``load_systems()`` / ``resolve_system_name()`` API.

v0.5.3 (2026-05-23)
-------------------

Added
^^^^^
* ISCro4 OOD gate probe (D2TGM5), a *Citrobacter rodentium* IS110-family bridge
  recombinase, verifying that the Tier-A IS110 gate fires for out-of-distribution
  IS110 proteins and returns ``DSB_FREE_TRANSEST_RECOMBINASE`` with
  ``tier_a_gate_override=True`` and confidence >= 0.90.
* ``tier_a_gate_override`` unit test.

Changed
^^^^^^^
* Bumped the ``genome-atlas`` pin to ``>=0.7.1,<0.8.0``, restoring
  SIMILAR_TO / HAS_RNA / PART_OF edges via ``graph_view='full'`` and adding
  ISCro4 (D2TGM5) to the atlas.

v0.5.2 (2026-05-22)
-------------------

Fixed
^^^^^
* Tier-A IS110 hard gate (``api.py``). IS110-family bridge recombinases were
  classified as DSB_NUCLEASE when scored without a pre-computed ESM-2 embedding
  (domain-only path): the Tier-A model was trained on IS110 proteins that all
  carried real ESM-2 embeddings, so a zero-seq domain vector is
  out-of-distribution. When PF01548 (DEDD_Tnp_IS110) and PF02371 (Transposase_20)
  are both present, ``tier_a`` is forced to ``DSB_FREE_TRANSEST_RECOMBINASE`` and
  ``tier_a_gate_override`` is set for the audit trail.

v0.5.1 (2026-05-11)
-------------------

Fixed
^^^^^
* Composite head domain gate (``api.py``). SpCas9 (Q99ZW2) fired
  ``composite=True`` at P=0.753 under the ML-only head (25% hold-out FP rate).
  The gate now requires PF01548 and PF02371 both present, forcing composite
  False otherwise. Composite FP rate: 0/4 = 0%, within the pre-registered
  <= 10% threshold.

v0.5.0 (2026-05-07)
-------------------

**First public pre-release.**

New features
^^^^^^^^^^^^
* ``Predictor.load()`` + ``predict_from_sequence()`` API for single-protein inference.
* ``Predictor.predict_from_fasta()`` for FASTA-file batch prediction.
* ``Predictor.predict_batch()`` for DataFrame batch prediction.
* ESM-2 150M singleton (lazy-loaded, CPU-only) for the F_seq channel.
* Pfam domain features (26-dim; F_domain channel) with UniProt REST fallback.
* Composite architecture flag (IS110 binary head; ``composite_prob`` field).
* Tier-B sub-class labels (supplementary; not gated in pre-registration).

Model performance
^^^^^^^^^^^^^^^^^
* Tier-A OOD holdout: 5/5 probes PASS (IS110, Fanzor, SpCas9, Bxb1, Tn5).
* Tier-A 5-fold CV macro-F1: 0.9862.
* IS110 reclassification: 31,870 of 31,871 (99.997%) of PF01548+PF02371 proteins
  correctly assigned DSB_FREE_TRANSEST_RECOMBINASE.

Known limitations
^^^^^^^^^^^^^^^^^
* SaProt structure channel (F_struct) zero-filled at inference (GPU/PDB required).
* Model artifact URL pending (``_download_models()`` stub).

v0.4.0 (internal)
------------------
Training script completion (scripts 01-24): label provenance, feature matrix,
model training, holdout validation.
