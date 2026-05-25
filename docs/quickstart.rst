Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install mech-class                    # core (LightGBM + domain features)
   pip install "mech-class[seq]"             # + ESM-2 150M (sequence embeddings)
   pip install "mech-class[seq,struct]"      # + SaProt 650M (structure, GPU required)

Loading trained models
----------------------

Trained model artifacts are provided as raw data files
(``mech-class v1.0``). Until then, point ``Predictor.load()`` at the
local model directory produced by the training scripts:

.. code-block:: python

   from mech_class.api import Predictor

   predictor = Predictor.load("/path/to/models")

Predicting from a sequence
--------------------------

.. code-block:: python

   pred = predictor.predict_from_sequence(
       accession="A0A7C9VKZ0",
       sequence="MKTAYIAK...",    # full amino acid sequence
       pfam_hits=["PF01548", "PF02371"],   # optional: skip UniProt REST lookup
   )

   print(pred.tier_a)               # "DSB_FREE_TRANSEST_RECOMBINASE"
   print(pred.tier_a_confidence)    # 0.997
   print(pred.composite)            # True
   print(pred.composite_prob)       # 0.999
   print(pred.summary())
   # A0A7C9VKZ0: DSB_FREE_TRANSEST_RECOMBINASE (conf=0.997) [COMPOSITE P=0.999]

Feature channels
^^^^^^^^^^^^^^^^

``channels_used`` reports which feature channels contributed non-zero signal:

* **F_seq** - ESM-2 150M mean-pool (640-dim). Requires ``pip install "mech-class[seq]"``.
  Auto-downloaded (~540 MB) on first call. Zero-filled if not installed.
* **F_domain** - Pfam binary flags (26-dim). Populated from ``pfam_hits``
  or fetched via UniProt REST API.
* **F_struct** - SaProt 650M structure embeddings (1280-dim). Zero-filled
  unless a PDB/CIF path is provided (``pdb_path=`` argument; GPU required).
* **F_active_site** - active-site geometry flags (7-dim). Zero-filled
  unless structure features are available.

Predicting from a FASTA file
-----------------------------

.. code-block:: python

   results = predictor.predict_from_fasta("my_enzymes.fasta")
   for pred in results:
       print(pred.summary())

Batch prediction from a DataFrame
----------------------------------

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({
       "accession": ["A0A7C9VKZ0", "Q8I6T1"],
       "sequence":  ["MKTAYIAK...", "MSTQVPG..."],
       "pfam_hits": [["PF01548", "PF02371"], ["PF07282"]],
   })
   results_df = predictor.predict_batch(df)
   print(results_df[["accession", "tier_a", "tier_a_confidence", "composite"]])

Understanding the output
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``tier_a``
     - str
     - Tier-A mechanism class: ``DSB_NUCLEASE``, ``DSB_FREE_TRANSEST_RECOMBINASE``,
       or ``TRANSPOSASE``
   * - ``tier_a_confidence``
     - float
     - Softmax probability of the predicted Tier-A class (0-1)
   * - ``tier_b``
     - str or None
     - Tier-B sub-class (e.g. ``N1_CRISPR_Cas``, ``B3_Programmable_Recombinase``).
       May be ``None`` if the sub-class model is not available or confidence < 0.5.
   * - ``composite``
     - bool
     - ``True`` if the protein is predicted to have a composite catalytic
       architecture (IS110-family canonical case: RuvC-fold DEDD + serine Tnp).
   * - ``composite_prob``
     - float
     - Probability of composite architecture from the binary head (0-1).
   * - ``pfam_hits``
     - list[str]
     - Pfam accessions used for the F_domain channel.
   * - ``channels_used``
     - list[str]
     - Feature channels that contributed non-zero signal.

IS110 composite architecture
----------------------------

The IS110 family (e.g. ``A0A7C9VKZ0``) encodes a bipartite catalytic
mechanism: an N-terminal RuvC-fold DEDD domain (PF01548) and a C-terminal
serine transposase domain (PF02371). Domain-only lookup (InterPro CL0219 clan)
systematically mis-classifies these as ``DSB_NUCLEASE``. The composite head
corrects this to ``DSB_FREE_TRANSEST_RECOMBINASE`` for 99.9% of the 31,870
IS110-family proteins in the GENOME-ATLAS catalog.

.. note::

   The composite head has a **25% false-positive rate** on the non-IS110 probe
   set (SpCas9 fires as composite=True, P=0.753). Treat the composite flag as a
   triage signal for IS110-like architecture, not a definitive classifier.
   See ``MODEL_CARD.md`` Limitation 3 for details.
