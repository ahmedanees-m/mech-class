# docs/conf.py — Sphinx configuration for mech-class
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Project metadata ──────────────────────────────────────────────────────────
project   = "mech-class"
author    = "Anees Ahmed"
copyright = "2024–2026, Anees Ahmed"

try:
    from mech_class._version import version as release
except ImportError:
    release = "0.5.3"
version = ".".join(release.split(".")[:2])

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy docstring style
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",              # Markdown (.md) file support
]

# ── Paths ─────────────────────────────────────────────────────────────────────
templates_path   = ["_templates"]
html_static_path = []           # empty: no _static content yet (avoids missing-dir warning)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme       = "furo"
html_title       = "mech-class"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/ahmedanees-m/mech-class/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# ── autodoc ───────────────────────────────────────────────────────────────────
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "private-members":  False,   # exclude _fetch_pfam_hits etc. (avoids ref warnings)
    "show-inheritance": True,
    "member-order":     "bysource",
}
autosummary_generate = True

# ── Napoleon (docstring style) ────────────────────────────────────────────────
napoleon_numpy_docstring        = True
napoleon_google_docstring       = False
napoleon_include_init_with_doc  = True

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
