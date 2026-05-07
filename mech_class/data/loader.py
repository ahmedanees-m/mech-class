"""Data loading helpers — taxonomy, labels, features."""

from __future__ import annotations

from importlib.resources import files as pkg_files
from pathlib import Path

import pandas as pd
import yaml


def load_taxonomy() -> dict:
    """Load the Tier A + Tier B class taxonomy."""
    raw = yaml.safe_load(pkg_files("mech_class").joinpath("data/label_taxonomy.yaml").read_text())
    return raw


def load_tier_a_classes() -> list[str]:
    """Return ordered list of Tier A class IDs."""
    return [c["id"] for c in load_taxonomy()["tier_a"]]


def load_tier_b_classes() -> list[str]:
    """Return ordered list of Tier B class IDs."""
    return [c["id"] for c in load_taxonomy()["tier_b"]]


def load_mechanism_labels(path: Path) -> pd.DataFrame:
    """Load the gold-standard mechanism labels parquet."""
    return pd.read_parquet(path)


def load_pfam_whitelist() -> list[dict]:
    """Load Pfam whitelist v1.2.0 from genome_atlas package data (Paper 1)."""
    raw = yaml.safe_load(pkg_files("genome_atlas").joinpath("data/pfam_whitelist.yaml").read_text())
    return raw["domains"]  # list of dicts: accession, name, mechanism_bucket, ...


def load_foundational_systems() -> list[dict]:
    """Load foundational systems v0.6.0 from genome_atlas package data (Paper 1)."""
    raw = yaml.safe_load(pkg_files("genome_atlas").joinpath("data/foundational_systems.yaml").read_text())
    return raw["systems"]  # list of dicts: name, mechanism_bucket, proteins[], ...
