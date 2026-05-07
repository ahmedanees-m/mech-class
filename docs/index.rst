mech-class documentation
========================

**mech-class** is a two-tier LightGBM classifier for mechanism prediction of
programmable genome-writing enzymes. It assigns each protein a Tier-A label
(DSB_NUCLEASE / DSB_FREE_TRANSEST_RECOMBINASE / TRANSPOSASE) and an optional
Tier-B sub-class label, using a 1953-dimensional feature vector combining
ESM-2 150M sequence embeddings, Pfam domain flags, and (optionally) SaProt
structure embeddings.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/predictor
   api/features
   api/models

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
