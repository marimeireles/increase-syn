#!/bin/bash
#SBATCH --job-name=rc-phiid-cpu
#SBATCH --account=def-zhijing
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/rc_phiid_cpu_%j.out
#SBATCH --error=slurm/logs/rc_phiid_cpu_%j.err

# PhiID + ranking (CPU-only, no GPU needed)

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /home/marimeir/scratch/increase-syn
mkdir -p slurm/logs

echo "=== Random Control: PhiID + Ranking (CPU-only) ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "--- Phases 2,3: PhiID + ranking ---"
echo "$(date)"
python scripts/run_pipeline.py \
    --model gemma3-4b-it-random-ctrl \
    --phases 2 3 \
    --max-workers 48

echo "=== Complete ==="
echo "End time: $(date)"
