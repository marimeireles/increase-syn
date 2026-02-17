#!/bin/bash
#SBATCH --job-name=gemma3it-phiid
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/gemma3it_base_%j.out
#SBATCH --error=slurm/logs/gemma3it_base_%j.err

# Gemma 3 4B-IT baseline synergy pipeline (Phase F0.5)
# Phase 1: ~1 hr (GPU), Phase 2: ~8-10 hrs (CPU), Phase 3: seconds
# Phase 4: ~4-8 hrs (GPU), Phase 6: seconds
# Total: ~12-20 hrs

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/marimeir/scratch/increase-syn
mkdir -p slurm/logs

echo "=== Gemma 3 4B-IT Baseline Synergy Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load secrets from .env (HF_TOKEN, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

python scripts/run_pipeline.py --model gemma3-4b-it --phases 1 2 3 4 6 --max-workers 32

echo "=== Complete ==="
echo "End time: $(date)"
