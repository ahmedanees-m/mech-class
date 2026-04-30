"""Pull UniProt ACT_SITE / BINDING feature evidence.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/03_pull_uniprot_features.py"

Expected output: /data/labels/evidence/uniprot_features.parquet
"""
from mech_class.evidence.uniprot_features import main

if __name__ == "__main__":
    main()
