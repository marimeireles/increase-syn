#!/bin/bash
#SBATCH --job-name=rc-phiid
#SBATCH --account=def-zhijing
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/rc_phiid_%j.out
#SBATCH --error=slurm/logs/rc_phiid_%j.err

# PhiID + ranking + ablation + comparison for random-control model
# Phase 1 (activations) already done — only runs phases 2,3,4,6 + comparison

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /home/marimeir/scratch/increase-syn
mkdir -p slurm/logs

echo "=== Random Control: PhiID + Ablation ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "--- Phases 2,3,4,6: PhiID, ranking, ablation, viz ---"
echo "$(date)"
python scripts/run_pipeline.py \
    --model gemma3-4b-it-random-ctrl \
    --phases 2 3 4 6 \
    --max-workers 32

echo "--- Three-way comparison ---"
echo "$(date)"
python scripts/25_compare_controls.py

echo "=== Complete ==="
echo "End time: $(date)"
