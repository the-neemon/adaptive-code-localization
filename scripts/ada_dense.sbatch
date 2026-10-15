#!/bin/bash
# Dense pass (acl.dense) on one ada 2080 Ti. Pass repos as arguments, e.g.
#   sbatch -J dense_django scripts/ada_dense.sbatch django/django
# With no arguments, every repository is processed. Resumable per task.
#SBATCH -A research
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o data/logs/%x_%j.log

set -euo pipefail
cd ~/ire/project/acl
nvidia-smi -L
args=()
for r in "$@"; do args+=(--repo "$r"); done
PYTHONPATH=src ~/miniconda3/envs/totto/bin/python -m acl.dense "${args[@]}" 2>&1 \
    | grep --line-buffered -vE "Warning|warnings.warn"
echo "DONE"
