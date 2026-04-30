"""Pull Rhea reactions evidence.

Run via:
    docker run --rm \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "pip install -e . --quiet && python scripts/02_pull_rhea.py"

Expected output: /data/labels/evidence/rhea.parquet
"""
from mech_class.evidence.rhea import main

if __name__ == "__main__":
    main()
