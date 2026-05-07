"""Step 4.5 — Pull TnPedia / ISfinder IS-family evidence (Week 2).

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/05_pull_tnpedia.py"

Inputs: /data/processed/isfinder.parquet (from Paper 1 deposit)
Expected output: /data/labels/evidence/tnpedia.parquet
"""
from mech_class.evidence.tnpedia import main

if __name__ == "__main__":
    main()
