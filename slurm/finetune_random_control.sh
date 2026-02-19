#!/bin/bash
#SBATCH --job-name=random-ctrl-ft
#SBATCH --account=def-zhijing
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/random_ctrl_%j.out
#SBATCH --error=slurm/logs/random_ctrl_%j.err

# Random-confidence control fine-tuning + synergy pipeline + comparison
#
# Phases:
#   20: Prepare random-confidence training data (~1 min, needs tokenizer/GPU)
#   21: LoRA fine-tuning (~2 hrs GPU)
#   22: Evaluate calibration (~30 min GPU)
#   23: Merge LoRA adapter (~5 min)
#   24: Synergy pipeline phases 1-4,6 (~12-20 hrs: 1hr GPU + 8-10hrs CPU + 4-8hrs GPU)
#   25: Three-way comparison (~seconds)
# Total: ~16-30 hrs
#
# This script is self-contained. Submit with:
#   sbatch slurm/finetune_random_control.sh

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/marimeir/scratch/increase-syn
mkdir -p slurm/logs results/rm-confounding-factors

echo "=== Random-Confidence Control Fine-tuning Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load secrets from .env (HF_TOKEN, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "--- Step 1: Prepare random-confidence training data ---"
echo "$(date)"
python scripts/20_prepare_random_control_data.py

echo "--- Step 2: LoRA fine-tuning (random confidence) ---"
echo "$(date)"
python scripts/21_finetune_random_control.py

echo "--- Step 3: Evaluate calibration ---"
echo "$(date)"
python scripts/22_evaluate_random_control.py

echo "--- Step 4: Merge LoRA adapter ---"
echo "$(date)"
python scripts/23_merge_random_control.py

echo "--- Step 5: Synergy pipeline on random-control model ---"
echo "$(date)"
python scripts/24_synergy_random_control.py

echo "--- Step 6: Three-way comparison ---"
echo "$(date)"
python scripts/25_compare_controls.py

echo "=== Complete ==="
echo "End time: $(date)"
