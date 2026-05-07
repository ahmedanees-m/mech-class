"""Step 4.9 — Atlas domain-based protein labelling (Week 2).

Joins ATLAS nodes_protein → HAS_DOMAIN edges → nodes_domain to produce
protein-level mechanism labels from the Pfam whitelist v1.2.0 and
InterPro clan annotations. Requires no external API calls.

Run via:
    docker run --rm \
        -v ~/pen-stack/data:/data \
        -v ~/pen-stack/code/repos/mech-class:/pkg \
        -w /pkg pen-stack/structure:0.1.0 \
        bash -c "pip install -e . --quiet && python scripts/05e_pull_atlas_domains.py"

Inputs:
    /data/graphs/atlas.duckdb
    /data/labels/evidence/interpro.parquet  (produced by step 04)
Expected output:
    /data/labels/evidence/atlas_domain_evidence.parquet  (~60k rows)
"""
from mech_class.evidence.atlas_domain import main

if __name__ == "__main__":
    main()
