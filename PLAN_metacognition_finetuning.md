# Plan: Metacognitive Fine-tuning of Gemma 3 4B-IT + Synergy Measurement

## Context

**Goal**: Fine-tune `google/gemma-3-4b-it` using the Steyvers et al. method (arxiv 2510.05126) to improve metacognition (calibrated confidence estimation), then measure whether the model's synergistic core changes using the existing PhiID pipeline.

**Scientific question**: Does improving a model's metacognitive abilities (calibrated uncertainty communication) alter the information-theoretic structure (synergy/redundancy) of its attention heads?

**Paper method (Steyvers et al.)**: Supervised fine-tuning with consistency-based confidence targets:
1. Sample model 10 times per question (temperature=1)
2. Consistency score = fraction matching modal answer
3. Target confidence = empirical_accuracy(consistency_bin) + noise
4. Two tasks: single-question confidence + pairwise comparison
5. Multi-task + multi-domain training gives best generalization

**OSF repo with paper data/code**: https://osf.io/k32wa/overview?view_only=f26ef0708a5d49ab8312beb8adfe3314

---

## Server Recommendation: Narval

Stay on Narval. Submitting multiple SLURM jobs is normal — the Gemma 12B experiment and this Gemma 4B fine-tuning run on separate nodes. The `syn` environment, HF cache, and phyid are already set up. A100-40GB is sufficient for Gemma 4B LoRA fine-tuning (~15GB VRAM peak). No need to rebuild everything on a different cluster.

If Narval queues become slow, Rorqual (H100s, replacing Beluga) is the best alternative — same SLURM ecosystem, faster GPUs, but would require environment setup from scratch.

---

## New Files to Create

```
rep-synergy-llm/
├── finetuning/                          # NEW: metacognition fine-tuning pipeline
│   ├── __init__.py
│   ├── config.py                        # Fine-tuning hyperparameters
│   ├── consistency_scorer.py            # Sample model N times, compute consistency
│   ├── semantic_equivalence.py          # DeBERTa NLI for TriviaQA answer matching
│   ├── prompt_templates.py              # Steyvers-style confidence prompt templates
│   ├── data_preparation.py              # Build training dataset
│   ├── train.py                         # LoRA fine-tuning with TRL SFTTrainer
│   ├── evaluate.py                      # ECE + AUC calibration evaluation
│   └── merge_adapter.py                 # Merge LoRA into base model
├── scripts/
│   ├── 10_prepare_finetune_data.py      # Phase F1
│   ├── 11_finetune_gemma.py             # Phase F2
│   ├── 12_evaluate_calibration.py       # Phase F3
│   ├── 13_merge_lora.py                 # Phase F4
│   ├── 14_synergy_finetuned.py          # Phase F5 (calls run_pipeline.py)
│   └── 15_compare_synergy.py            # Phase F6 (comparison figures)
├── slurm/
│   ├── finetune_gemma3.sh               # SLURM: fine-tuning (F1-F4, ~10 hrs)
│   └── synergy_gemma3_ft.sh             # SLURM: synergy pipeline (F5-F6, ~24 hrs)
└── results/finetuning/                  # NEW: output directory
    ├── training_data/
    ├── checkpoints/
    ├── merged_model/
    └── eval/
```

## Existing Files to Modify

- **`configs/config.py`**: Add `gemma3-4b-it-ft` to `MODEL_CONFIGS` (base model already exists)
- **`scripts/run_pipeline.py`**: Add `gemma3-4b-it-ft` to `_infer_architecture()` known dict and argparse `choices`

---

## Phase F0: Environment Setup (Login Node — needs internet)

```bash
micromamba activate syn
pip install peft trl bitsandbytes datasets scikit-learn
```

Pre-download (login node only):
- `google/gemma-3-4b-it` (HF_TOKEN required, gated model)
- `microsoft/deberta-v3-large-mnli` (for TriviaQA semantic equivalence)
- Datasets: `TIGER-Lab/MMLU-Pro`, `openai/gsm8k`, `trivia_qa` (rc.nocontext)

---

## Phase F1: Generate Consistency Targets (~4-6 hrs GPU)

**Script**: `scripts/10_prepare_finetune_data.py`
**Core modules**: `finetuning/consistency_scorer.py`, `finetuning/semantic_equivalence.py`, `finetuning/data_preparation.py`

For each training question (2000 MMLU-Pro + 2000 GSM8K + 800 TriviaQA):
1. Sample Gemma 3 4B-IT **10 times** at temperature=1
2. Extract answers:
   - MMLU-Pro: regex for letter choice (A-J)
   - GSM8K: regex for `#### <number>` pattern
   - TriviaQA: full text answer
3. Compute consistency: fraction of 10 samples matching modal answer
4. Determine correctness: compare modal answer to ground truth
   - MMLU-Pro/GSM8K: exact match
   - TriviaQA: first try exact match against all aliases (case-insensitive), then fall back to DeBERTa-v3-large-MNLI bidirectional entailment (replaces GPT-4o judge from paper)
5. Compute empirical accuracy per consistency bin (10 bins)
6. Assign target confidence: `accuracy(bin) + ε`, where `ε ~ Uniform(-0.05, 0.05)`, clamped to [0, 1]
7. Balance distribution: subsample overrepresented consistency bins
8. Generate training examples for both tasks:
   - **Single-question**: prompt with "Answer: \nConfidence score (0-1):" format, target = answer + confidence
   - **Pairwise**: pair questions from different consistency bins, target = which has higher confidence + both answers
9. Format as chat conversations for Gemma IT tokenizer (`apply_chat_template`)

**Optimization**: Batch generation (batch_size=16), reduce max_new_tokens to 64 for MCQ, 128 for math/trivia.

**Output**: HuggingFace Dataset saved to `results/finetuning/training_data/`

---

## Phase F2: LoRA Fine-tuning (~2 hrs GPU)

**Script**: `scripts/11_finetune_gemma.py`
**Core module**: `finetuning/train.py`

**LoRA config**:
- `r=16`, `alpha=32` (scaling = 2)
- `dropout=0.05`
- `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` — attention projections ONLY
- **Why attention-only**: The synergy pipeline measures attention head L2 norms. Targeting only attention projections means any synergy change is directly attributable to how attention heads process information, not confounded by MLP changes.

**Training config**:
- `epochs=5` (matches paper's Llama config)
- `batch_size=4`, `gradient_accumulation=8` (effective batch=32)
- `lr=2e-4`, cosine schedule, warmup_ratio=0.03
- `bf16=True`, gradient checkpointing
- `max_seq_length=512`

**Memory**: Model bf16 (~8GB) + LoRA (~50MB) + optimizer (~200MB) + activations (~4GB) = ~15GB. Well within A100-40GB.

**Output**: LoRA adapter at `results/finetuning/checkpoints/`

---

## Phase F3: Evaluate Calibration (~1 hr GPU)

**Script**: `scripts/12_evaluate_calibration.py`
**Core module**: `finetuning/evaluate.py`

Evaluate both base and fine-tuned models on held-out test sets (15% of each dataset):

**Metrics**:
- **ECE** (Expected Calibration Error): `ECE = Σ (|B_m|/N) × |acc(B_m) - conf(B_m)|` with 10 bins. Lower = better.
- **AUC** (discrimination): probability that correct answers get higher confidence than incorrect. Higher = better.
- Parse success rate (fraction of outputs with extractable confidence scores)

**Output**: JSON files at `results/finetuning/eval/` with per-domain and aggregate metrics.

---

## Phase F4: Merge LoRA Adapter (~5 min)

**Script**: `scripts/13_merge_lora.py`
**Core module**: `finetuning/merge_adapter.py`

Merge LoRA weights into base model using PEFT's `merge_and_unload()`. Save full model to `results/finetuning/merged_model/`. This ensures the synergy pipeline works identically — no PEFT dependency at inference, same hook behavior, clean integration.

**Output**: Merged model (~8GB) at `results/finetuning/merged_model/`

---

## Phase F5: Synergy Pipeline on Fine-tuned Model Only (~10-12 hrs)

**Script**: `scripts/14_synergy_finetuned.py` (thin wrapper calling `run_pipeline.py`)

**Note**: The base `google/gemma-3-4b-it` synergy data already exists from a previous run. We only need to run the pipeline on the fine-tuned model. Phase F6 loads the existing base results for comparison.

Integration with existing pipeline requires two small changes:

### 1. Add to `configs/config.py` MODEL_CONFIGS:
```python
"gemma3-4b-it-ft": {
    "model_name": "results/finetuning/merged_model",  # local path
    "model_id": "gemma3_4b_it_ft",
    "torch_dtype": "bfloat16",
},
```

### 2. Update `scripts/run_pipeline.py`:
- Add to `_infer_architecture()` known dict: `"gemma3_4b_it_ft": (8, 34)`
- Add to argparse choices: `"gemma3-4b-it-ft"`

Then run the existing pipeline on the fine-tuned model only:
```bash
python scripts/run_pipeline.py --model gemma3-4b-it-ft --phases 1 2 3 4 6 --max-workers 32
```

The model_registry's `detect_model_spec()` already handles Gemma 3 (model_type `gemma3_text`/`gemma3`) — the merged model has the same config.json. `load_model_and_tokenizer()` already supports local paths.

**Output**: Full synergy results (activations, PhiID matrices, rankings, ablation, figures) for the fine-tuned model in `results/`. Base model results already exist.

---

## Phase F6: Compare Synergy Profiles

**Script**: `scripts/15_compare_synergy.py`

Generate comparison figures:
1. **Overlaid PhiID profiles**: base vs fine-tuned syn-red score by layer depth (does the inverted-U shift?)
2. **Delta heatmap**: per-head syn_red_score change (layers × heads). Where do changes concentrate?
3. **Ablation comparison**: does synergistic-first ablation cause faster/slower divergence after fine-tuning?
4. **Statistics**: paired t-test, Wilcoxon test, effect size (Cohen's d) on per-layer means. Spearman correlation between base and fine-tuned head rankings.

**Output**: Figures at `results/figures/`

---

## SLURM Jobs

### Job 1: Fine-tuning (`slurm/finetune_gemma3.sh`)
- `--time=24:00:00`, `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=48G`
- Runs phases F0-F4 sequentially
- ~10 hrs total (F1: 4-6 hrs, F2: 2 hrs, F3: 1 hr, F4: 5 min)

### Job 2: Synergy analysis (`slurm/synergy_gemma3_ft.sh`)
- `--time=24:00:00`, `--gres=gpu:1`, `--cpus-per-task=32`, `--mem=48G`
- `--dependency=afterok:<job1_id>`
- Runs fine-tuned model only through synergy pipeline + comparison with existing base results
- ~10-12 hrs total (one model only; base model data already exists)
- PhiID has checkpoint/resume — safe to resubmit if timeout

---

## Verification Checklist

1. **F1**: Training dataset size ~3500-4000 after balancing. Consistency distribution roughly uniform. All confidences in [0,1]. Spot-check 10 examples per dataset.
2. **F2**: Training loss decreases. Eval loss plateaus (no severe overfitting). ~10-20M trainable params. Sample outputs produce confidence scores.
3. **F3**: ECE lower and AUC higher for fine-tuned vs base. Parse success rate >90%.
4. **F4**: Merged model produces same outputs as base+adapter. `detect_model_spec()` reports correct architecture (8 heads, 34 layers).
5. **F5**: Activation shapes = (60, 272, 100). PhiID matrices = (272, 272). Fine-tuned model shows some inverted-U profile (compare against existing base results).
6. **F6**: Statistical test p-values reported. Head ranking correlation between models reported.

---

## Expected Outcomes

- **Hypothesis**: Metacognition fine-tuning may slightly increase synergy in middle layers (more integrated processing for calibrated confidence)
- **Alternative**: Profiles are nearly identical — metacognition is implemented through parameter changes that don't affect PhiID-measured structure
- **Either result is informative**: positive change = metacognition requires synergistic processing; null = metacognition is orthogonal to the synergistic core
