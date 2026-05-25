# MECH-CLASS Update Strategy

Documents how the model and label set are maintained across versions.

## Version scheme

`MAJOR.MINOR.PATCH` - follows the same convention as GENOME-ATLAS.

| Change type | Version bump | Action required |
|---|---|---|
| New training examples added (<=20% increase) | MINOR | Re-train, re-run bootstrap CIs, update MODEL_CARD |
| New Tier-B class added | MINOR | Re-register pre-trained criteria for new class only |
| New feature channel added | MINOR | Full ablation re-run required |
| IS110 composite rule changed | MINOR | Re-run all IS110 holdout probes |
| Pre-registered success criteria changed | MAJOR | New bioRxiv preprint section required |
| Tier-A class definitions changed | MAJOR | Full re-training + re-registration |

## Trigger conditions for a v1.x update

- New curated IS110-family proteins added to TnPedia/TnCentral after release date
- New Fanzor/OMEGA ortholog families discovered (Saito et al. follow-up, EVOLVEpro updates)
- AlphaFold database major version update (v4->v5) changes structure quality for >5% of training proteins
- Any training protein is retracted or found to be misannotated in its source database

## Data freshness policy

| Source | Check frequency | Staleness threshold |
|---|---|---|
| M-CSA | Monthly | >6 months stale triggers review |
| TnPedia | Quarterly | >12 months stale triggers review |
| CRISPRCasdb | Quarterly | Major version release triggers update |
| UniProt SwissProt | At each GENOME-ATLAS atlas update | Tied to GENOME-ATLAS release cycle |
| AlphaFold | At each GENOME-ATLAS atlas update | Tied to GENOME-ATLAS release cycle |

## Compatibility with GENOME-ATLAS

`mech-class` declares `genome-atlas>=0.6.0`. Any GENOME-ATLAS release that breaks the
embeddings table schema or renames the `atlas.duckdb` tables will require a mech-class
compatibility patch. GENOME-ATLAS follows semantic versioning; breaking changes are MAJOR.

## Backward compatibility

- Trained models in `/data/models/tier_a/model.lgb` are versioned by date stamp in filename
- Old models are retained for 6 months after a new version releases
- The `Predictor.load()` API is stable across all MINOR versions

## Deprecation notice

When IS110 composite detection is eventually superseded by a learned end-to-end head
(requires >50 IS110 training examples), the rule-based override will be deprecated with
one MINOR version notice before removal.
