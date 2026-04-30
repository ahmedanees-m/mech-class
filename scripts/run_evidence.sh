#!/usr/bin/env bash
# Evidence ingestion pipeline (scripts 01 through 06).
# Run: bash scripts/run_evidence.sh
#
# Prerequisites:
#   - the structure image built from containers/structure/
#   - GENOME-ATLAS artifacts under $DATA
#   - a local checkout of the genome-atlas repo
#
# Outputs evidence parquets under $DATA/labels/evidence/; continue with
# scripts/07_review_queue.py then scripts/08_ingest_curator_decisions.py.

set -euo pipefail

MECH_CLASS=~/pen-stack/code/repos/mech-class
GENOME_ATLAS=~/pen-stack/code/repos/genome-atlas
DATA=~/pen-stack/data

DOCKER_BASE="docker run --rm \
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
  -v ${DATA}:/data \
  -v ${MECH_CLASS}:/pkg \
  -v ${GENOME_ATLAS}:/genome-atlas \
  -w /pkg pen-stack/structure:0.1.0"

SETUP='git config --global --add safe.directory /pkg && \
       git config --global --add safe.directory /genome-atlas && \
       pip install -e /genome-atlas --quiet && \
       pip install -e . --quiet'

echo "=== Evidence ingestion ==="
echo ""

echo "--- Step 01: M-CSA ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/01_pull_mcsa.py"

echo "--- Step 02: Rhea (2-step lookup) ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/02_pull_rhea.py"

echo "--- Step 03: UniProt features ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/03_pull_uniprot_features.py"

echo "--- Step 04: InterPro clans ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/04_pull_interpro.py"

echo "--- Step 05: TnPedia / ISfinder ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/05_pull_tnpedia.py"

echo "--- Step 05b: CRISPRCasdb (GENOME-ATLAS deposit) ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/05b_pull_crisprcasdb.py"

echo "--- Step 05c: Pfam whitelist ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/05c_pull_pfam_whitelist.py"

echo "--- Step 05d: Foundational systems ---"
${DOCKER_BASE} bash -c "${SETUP} && \
  python scripts/05d_pull_foundational.py"

echo "--- Step 06: Aggregate evidence ---"
${DOCKER_BASE} bash -c "${SETUP} && python scripts/06_aggregate_evidence.py"

echo ""
echo "=== Evidence files written to ${DATA}/labels/evidence/ ==="
ls -lh ${DATA}/labels/evidence/*.parquet 2>/dev/null || echo "(no parquets found)"
echo ""
echo "=== Aggregated labels ==="
ls -lh ${DATA}/labels/mechanism_labels_raw.parquet 2>/dev/null || true
ls -lh ${DATA}/labels/review_queue/ 2>/dev/null || true
echo ""
echo "=== Done. Inspect review_queue.parquet and run script 07 next. ==="
