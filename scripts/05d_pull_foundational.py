"""Foundational systems anchor labels from GENOME-ATLAS.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/05d_pull_foundational.py"

Expected output: /data/labels/evidence/foundational.parquet (~25-30 rows)
Includes IS621 composite case assertion.
"""
from mech_class.evidence.foundational import main

if __name__ == "__main__":
    main()
