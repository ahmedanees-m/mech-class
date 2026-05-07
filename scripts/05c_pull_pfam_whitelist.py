"""Step 4.7 — Pfam whitelist v1.2.0 evidence from Paper 1 (Week 2).

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/05c_pull_pfam_whitelist.py"

Expected output: /data/labels/evidence/pfam_whitelist.parquet
"""
from mech_class.evidence.pfam_whitelist import main

if __name__ == "__main__":
    main()
