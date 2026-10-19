#!/bin/bash
# Phase A on an ada node (CPU only; needs data/prepared, data/dense, data/line_counts.json).
#   sbatch scripts/ada_phase_a.sbatch
# 1. tune the fusion RRF k on dev (expand settings are already in configs/tuned.json)
# 2. run every action on every task at B = 100/300/500
# 3. post-seed features, with BM25/dense agreement
# Copy data/results/ and configs/tuned.json back afterwards.
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -J phase_a
#SBATCH -o data/logs/%x_%j.log

set -euo pipefail
cd ~/ire/project/acl
PY="env PYTHONPATH=src $HOME/miniconda3/envs/totto/bin/python"
$PY -m acl.tune --what rrf
$PY -m acl.phase_a
$PY -m acl.features
echo "DONE"
