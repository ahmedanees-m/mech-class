# Changelog

All notable changes to `mech-class` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
