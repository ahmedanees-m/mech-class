"""Step 4.5b — Pull CRISPRCasdb evidence from Paper 1 deposit (Week 2).

Reuses crisprcasdb_systems.parquet and crisprcasdb_proteins.parquet from Paper 1.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/05b_pull_crisprcasdb.py"

Inputs: /data/processed/crisprcasdb_systems.parquet, crisprcasdb_proteins.parquet
Expected output: /data/labels/evidence/crisprcasdb.parquet
"""
from mech_class.evidence.crisprcasdb import main

if __name__ == "__main__":
    main()
