"""Step 4.4 — Pull InterPro clan inheritance evidence (Week 2).

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/04_pull_interpro.py"

Expected output: /data/labels/evidence/interpro.parquet
"""
from mech_class.evidence.interpro import main

if __name__ == "__main__":
    main()
