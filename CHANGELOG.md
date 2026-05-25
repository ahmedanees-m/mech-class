# Changelog

All notable changes to `mech-class` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.4] — 2026-05-25

### Changed
- **Renamed holdout probe** `ISCro4_IS622_OOD_gate_probe` → `ISCro4_pelea_2026` across
  all files (`holdout_set.yaml`, `holdout_results.json`, `test_predictor_api.py`).
  Canonical name per UniProt D2TGM5 + Pelea 2026 *Science* adz1884. "IS622" was a
  deprecated preprint label (Perry 2025 *bioRxiv* 2025.05.14.653916); retained in
  `aliases` fields for backward compatibility. 6/6 OOD holdout still passes — no model
  behaviour change, rename only.
- **`genome-atlas` pin bumped** (`pyproject.toml`). `atlas` optional extra updated from
  `>=0.7.1,<0.8.0` to `>=0.7.2,<0.8.0`. genome-atlas v0.7.2 adds canonical ISCro4
  naming (was IS622) with alias-resolution and `load_systems()` / `resolve_system_name()`
  API required by PEN-COMPARE v3.2.

### Compatibility
- Required by **PEN-COMPARE v3.2** (ISCro4 canonical naming across all 4 PEN-STACK packages).
- Backward compatible — all API signatures unchanged; no model retraining.

## [0.5.3] — 2026-05-23

### Added
- **ISCro4 OOD holdout probe (D2TGM5)** (`tests/integration/test_predictor_api.py`,
  `tests/regression/test_holdout_probes.py`). Citrobacter rodentium ICC168 IS110-family
  bridge recombinase; highest-profile IS110 human-cell genome-writing result
  (Pelea 2026 *Science* doi:10.1126/science.adz1884; Perry 2025 bioRxiv 2025.05.14.653916).
  Verifies that the v0.5.2 Tier-A IS110 gate fires for truly OOD IS110 proteins
  (no pre-computed ESM-2 embedding) and returns `DSB_FREE_TRANSEST_RECOMBINASE` with
  `tier_a_gate_override=True`, confidence ≥ 0.90. n_ood: 5 → 6.
- **`tier_a_gate_override` unit test** (`tests/unit/test_api_helpers.py`). Two new
  assertions: field defaults to `False`; set to `True` when gate fires; present in
  `Prediction.model_dump()`.

### Changed
- **`genome-atlas` pin bumped** (`pyproject.toml`). `atlas` optional extra updated from
  `genome-atlas>=0.6.0,<0.7.0` to `>=0.7.1,<0.8.0`. v0.7.1 restores SIMILAR_TO/HAS_RNA/
  PART_OF edges via `graph_view='full'` (fixes atlas_domain feature extractor; v0.7.0 broke
  these edges), adds ISCro4 (D2TGM5) to the atlas, and achieves AUROC 0.9714 (GraphSAGE)
  with 41/41 tests passing. v0.7.0 is skipped — it broke SIMILAR_TO and adds no new
  mech-class training data.

## [0.5.2] — 2026-05-22

### Fixed
- **CRITICAL: Tier-A IS110 hard gate** (`api.py`). IS110-family bridge recombinases
  (e.g. IS621) were misclassified as DSB_NUCLEASE when scored at inference time without
  a pre-computed ESM-2 embedding (domain-only path). Root cause: the LightGBM Tier-A
  model was trained where all 14 IS110 proteins had real ESM-2 embeddings; a zero-seq
  + dom_4/dom_5 feature vector is OOD and the model incorrectly fires DSB_NUCLEASE.
  Fix: biochemical hard gate — if PF01548 (DEDD_Tnp_IS110) AND PF02371 (Transposase_20)
  are both present, `tier_a` is forced to `DSB_FREE_TRANSEST_RECOMBINASE` regardless of
  the ML head output. `tier_a_gate_override: bool` added to `Prediction` for audit trail.
  Confidence is set to max(ML_DSB_FREE_prob, 0.90). Same gate condition as composite head.
  IS110 training proteins score correctly without the gate when ESM-2 is available; the
  gate only fires in the OOD domain-only scenario.
- **`genome-atlas` upper-bound pin** (`pyproject.toml`). Pinned `atlas` optional extra to
  `genome-atlas>=0.6.0,<0.7.0`. v0.7.0 removes SIMILAR_TO edges (breaks atlas_domain
  feature extractor); all 3 new v0.7.0 YAML proteins already in 572-protein training set.

## [0.5.1] — 2026-05-11

### Fixed
- **Composite head FP gate** (`api.py`). SpCas9 (Q99ZW2) triggered `composite=True`
  at P=0.753 under the pure ML head. Pre-registered ≤10% FP criterion: FAIL (25%).
  Fix: biochemical hard gate requiring PF01548 AND PF02371 both present; composite forced
  False otherwise. `ml_composite_prob_raw` field added for audit trail. FP rate: 0/4 = 0%.
  Pre-registered criterion: PASS.

## [0.5.0] — 2026-04-30

### Added
- Full package scaffold under `PAPER_2/mech-class/`
- `mech_class/data/label_taxonomy.yaml` — Tier A + Tier B class taxonomy v1.0.0
- `mech_class/evidence/` — 8 evidence source modules (M-CSA, Rhea, UniProt features,
  InterPro, CRISPRCasdb, TnPedia, Pfam whitelist, foundational systems)
- `mech_class/evidence/aggregator.py` — weighted evidence aggregation → EvidenceRecord
- `mech_class/features/` — F_seq (ESM-2 reuse), F_struct (SaProt 650M), F_domain, F_active_site
- `mech_class/models/` — LightGBM Tier-A, Tier-B, composite flag head
- `mech_class/api.py` — public `Predictor` class
- All numbered pipeline scripts 00–41
- Test suite (unit, integration, regression)
- Documentation: LABEL_PROVENANCE.md, MODEL_CARD.md, UPDATE_STRATEGY.md, VALIDATION.md
- Docker: `containers/structure/Dockerfile` (pen-stack/structure:0.1.0)
- Pre-registered success criteria locked in label_taxonomy.yaml

## [0.0.1] — 2026-04-22

### Added
- Initial repository scaffold (GitHub: ahmedanees-m/mech-class)
- `pyproject.toml`, `LICENSE` (MIT), `README.md`, `CHANGELOG.md`, `CITATION.cff`
- `mech_class/__init__.py`, `_version.py`, `cli.py`
- `.github/workflows/ci.yml`, `docs.yml`
