#!/bin/bash
# Stage 1 prepare (acl.prepare) on an ada node (CPU only). Resumable per task.
#   sbatch scripts/ada_prepare.sbatch
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 20
#SBATCH --mem=36G
#SBATCH -t 06:00:00
#SBATCH -J prepare
#SBATCH -o data/logs/%x_%j.log

set -euo pipefail
cd ~/ire/project/acl
PYTHONPATH=src ~/miniconda3/envs/totto/bin/python -m acl.prepare --workers 18 2>&1 \
    | grep --line-buffered -v SyntaxWarning
echo "DONE"
