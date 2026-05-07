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
    release = "0.5.0"
version = ".".join(release.split(".")[:2])

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # Google/NumPy docstring style
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_nb",                      # Jupyter notebook rendering
]

# ── Paths ─────────────────────────────────────────────────────────────────────
templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme       = "furo"
html_title       = "mech-class"
html_static_path = ["_static"]

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
    "show-inheritance": True,
    "member-order":     "bysource",
}
autosummary_generate = True

# ── Napoleon (docstring style) ────────────────────────────────────────────────
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False
napoleon_include_init_with_doc = True

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3.10", None),
    "numpy":    ("https://numpy.org/doc/stable/", None),
    "pandas":   ("https://pandas.pydata.org/docs/", None),
    "sklearn":  ("https://scikit-learn.org/stable/", None),
}

# ── myst-nb ───────────────────────────────────────────────────────────────────
nb_execution_mode = "off"   # don't execute notebooks at build time (no GPU/models)
myst_enable_extensions = ["colon_fence", "deflist"]
