"""Step 4.1 — Pull M-CSA evidence (Week 2).

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/01_pull_mcsa.py"

Expected output: /data/labels/evidence/mcsa.parquet (~5,000 rows)
"""
from mech_class.evidence.mcsa import main

if __name__ == "__main__":
    main()
