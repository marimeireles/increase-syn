#!/bin/bash
#SBATCH --job-name=gemma3it-finetune
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/finetune_%j.out
#SBATCH --error=slurm/logs/finetune_%j.err

# Metacognitive fine-tuning pipeline (Phases F1-F4)
# F1: Prepare finetune data (~7 hrs GPU — consistency scoring)
# F2: LoRA fine-tuning (~2 hrs GPU)
# F3: Evaluate calibration (~30 min GPU)
# F4: Merge LoRA adapter (~5 min)
# Total: ~10 hrs
#
# Submit AFTER synergy_gemma3_it_base.sh completes:
#   sbatch --dependency=afterok:<job0_id> slurm/finetune_gemma3_it.sh

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/marimeir/scratch/increase-syn
mkdir -p slurm/logs results/finetuning

echo "=== Gemma 3 4B-IT Fine-tuning Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load secrets from .env (HF_TOKEN, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "--- Phase F1: Prepare fine-tuning data ---"
echo "$(date)"
python scripts/10_prepare_finetune_data.py

echo "--- Phase F2: LoRA fine-tuning ---"
echo "$(date)"
python scripts/11_finetune_gemma.py

echo "--- Phase F3: Evaluate calibration ---"
echo "$(date)"
python scripts/12_evaluate_calibration.py

echo "--- Phase F4: Merge LoRA adapter ---"
echo "$(date)"
python scripts/13_merge_lora.py

echo "=== Complete ==="
echo "End time: $(date)"
