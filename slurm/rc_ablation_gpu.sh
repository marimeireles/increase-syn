#!/bin/bash
#SBATCH --job-name=rc-ablation
#SBATCH --account=def-zhijing
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/rc_ablation_%j.out
#SBATCH --error=slurm/logs/rc_ablation_%j.err

# Ablation + viz + comparison (needs GPU)

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /home/marimeir/scratch/increase-syn
mkdir -p slurm/logs

echo "=== Random Control: Ablation + Comparison (GPU) ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "--- Phases 4,6: ablation + viz ---"
echo "$(date)"
python scripts/run_pipeline.py \
    --model gemma3-4b-it-random-ctrl \
    --phases 4 6 \
    --max-workers 8

echo "--- Three-way comparison ---"
echo "$(date)"
python scripts/25_compare_controls.py

echo "=== Complete ==="
echo "End time: $(date)"
