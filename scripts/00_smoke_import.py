"""Smoke test: verify genome-atlas is importable and GENOME-ATLAS artifacts load.

Run via:
    docker run --rm \\
        -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \\
        -v ~/pen-stack/data:/data \\
        -v ~/pen-stack/code/repos/mech-class:/pkg \\
        -v ~/pen-stack/code/repos/genome-atlas:/genome-atlas \\
        -w /pkg pen-stack/structure:0.1.0 \\
        bash -c "git config --global --add safe.directory /pkg && \\
                 git config --global --add safe.directory /genome-atlas && \\
                 pip install -e /genome-atlas --quiet && \\
                 pip install -e . --quiet && \\
                 python scripts/00_smoke_import.py"
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml
import duckdb
import pandas as pd
from importlib.resources import files as pkg_files

# Verify genome_atlas imports
print("Importing genome_atlas...")
import genome_atlas
print(f"  genome_atlas {genome_atlas.__version__} imported OK")

from genome_atlas.api import Atlas   # Atlas lives in api.py, not __init__

# Verify Pfam whitelist YAML
print("\nLoading Pfam whitelist v1.2.0...")
wl = yaml.safe_load(
    pkg_files("genome_atlas").joinpath("data/pfam_whitelist.yaml").read_text()
)
domains = wl["domains"]
print(f"  {len(domains)} primary domains")
for d in domains[:3]:
    print(f"    {d['accession']}  {d['name']}  -> {d['mechanism_bucket']}")

# Verify foundational systems YAML
print("\nLoading foundational systems...")
fs = yaml.safe_load(
    pkg_files("genome_atlas").joinpath("data/foundational_systems.yaml").read_text()
)
systems = fs["systems"]
print(f"  {len(systems)} systems")
for s in systems[:3]:
    print(f"    {s['name']}  ({s['mechanism_bucket']})")

# Verify IS621 composite case annotation
is621 = next((s for s in systems if "IS621" in s["name"]), None)
assert is621 is not None, "IS621 not found in foundational systems!"
assert is621["mechanism_bucket"] == "DSB_FREE_TRANSEST_RECOMBINASE", (
    f"IS621 mechanism_bucket is '{is621['mechanism_bucket']}', expected "
    f"'DSB_FREE_TRANSEST_RECOMBINASE' (GENOME-ATLAS correction)"
)
print(f"  IS621 composite case: mechanism_bucket = {is621['mechanism_bucket']} [OK]")

# Verify ATLAS DuckDB
print("\nLoading ATLAS DuckDB...")
con = duckdb.connect("/data/graphs/atlas.duckdb", read_only=True)
n_proteins = con.execute("SELECT COUNT(*) FROM nodes_protein").fetchone()[0]
n_domains  = con.execute("SELECT COUNT(*) FROM nodes_domain").fetchone()[0]
n_edges    = con.execute("SELECT COUNT(*) FROM edges WHERE edge_type='HAS_DOMAIN'").fetchone()[0]
con.close()
print(f"  {n_proteins:,} proteins, {n_domains} domains, {n_edges:,} HAS_DOMAIN edges")
assert n_proteins >= 9000, f"Expected >=9000 proteins, got {n_proteins}"
assert n_domains  >= 18,   f"Expected >=18 domains, got {n_domains}"

# Verify Atlas object (gpickle + embeddings - no db_path)
print("\nLoading ATLAS gpickle...")
atlas = Atlas.load(
    gpickle_path=Path("/data/graphs/atlas.gpickle"),
    embeddings_path=Path("/data/embeddings/graphsage_v6.parquet"),
    targets_path=Path("/data/processed/targets_v2_with_negatives.parquet"),
)
print("  Atlas loaded OK")

# Verify ESM-2 embeddings
print("\nLoading ESM-2 embeddings...")
esm = pd.read_parquet("/data/embeddings/esm2_150M_v6.parquet")
emb_dim = len(esm["embedding"].iloc[0])
print(f"  {len(esm):,} proteins x {emb_dim} dim")
assert emb_dim == 640, f"Expected 640-dim ESM-2 embeddings, got {emb_dim}"

# Verify mech-class taxonomy
print("\nLoading mech-class label_taxonomy.yaml...")
from mech_class.data.loader import load_taxonomy, load_tier_a_classes
tax = load_taxonomy()
tier_a = load_tier_a_classes()
print(f"  Tier A classes: {tier_a}")
assert "DSB_NUCLEASE" in tier_a
assert "DSB_FREE_TRANSEST_RECOMBINASE" in tier_a
assert "TRANSPOSASE" in tier_a

print("\n=== All checks passed. GENOME-ATLAS artifacts accessible. mech-class can build. ===")
