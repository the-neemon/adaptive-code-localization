#!/bin/bash
# Stage 5 on ada: the 90 extra SWE-bench Verified issues end to end (snapshots,
# chunks, BM25, graph, dense, all six actions, features), then the held-out
# router evaluation over everything.
#   sbatch scripts/ada_stage5.sbatch
# Needs the Phase A outputs of the 500 study issues (data/results/phase_a.jsonl,
# features.jsonl) and the dense embedding caches in data/dense/.
# Copy back data/results/heldout.md, heldout.json and data/results/verified_extra/.
#SBATCH -A research
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH -x gnode066,gnode068,gnode069,gnode076,gnode077
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J stage5
#SBATCH -o data/logs/%x_%j.log

set -euo pipefail
cd ~/ire/project/acl
PY="env PYTHONPATH=src $HOME/miniconda3/envs/totto/bin/python"
X=data/tasks_verified_extra.jsonl
nvidia-smi -L
# Some nodes report a GPU that torch cannot use; never let the dense pass run on CPU.
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this node'"

$PY -m acl.data --verified-extra
$PY -m acl.snapshot --tasks $X
$PY -m acl.prepare --tasks $X --workers 6 2>&1 | grep --line-buffered -v SyntaxWarning
$PY -m acl.dense --tasks $X 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"
mkdir -p data/results/verified_extra
$PY -m acl.phase_a --tasks $X --out data/results/verified_extra/phase_a.jsonl
$PY -m acl.features --tasks $X --out data/results/verified_extra/features.jsonl
$PY -m acl.heldout
echo "DONE"
