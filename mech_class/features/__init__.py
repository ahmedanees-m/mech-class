"""Feature channel modules for mechanism classification.

Four channels:
    F_seq  (640-dim)  — ESM-2 150M embeddings from Paper 1 (reuse, no re-inference)
    F_struct (1280-dim) — SaProt 650M structure-aware embeddings
    F_domain (~40-dim)  — Pfam binary presence + IS110 composite flag
    F_active_site (~20-dim) — Active-site Cα distances + DSSP (pLDDT ≥ 70 gated)
"""
