"""Configuration for metacognitive fine-tuning (Steyvers et al. method)."""

FT_CONFIG = {
    "base_model": "google/gemma-3-4b-it",
    "datasets": {
        "mmlu_pro": {
            "name": "TIGER-Lab/MMLU-Pro",
            "train_size": 500,
            "test_size": 250,
        },
        "gsm8k": {
            "name": "openai/gsm8k",
            "config": "main",
            "train_size": 500,
            "test_size": 250,
        },
        "trivia_qa": {
            "name": "mandarjoshi/trivia_qa",
            "config": "rc.nocontext",
            "train_size": 200,
            "test_size": 100,
        },
    },
    "num_samples": 10,          # samples per question for consistency
    "temperature": 1.0,
    "max_new_tokens_mcq": 64,
    "max_new_tokens_math": 128,
    "max_new_tokens_trivia": 64,
    "batch_size": 8,
    "noise_range": 0.05,        # epsilon for confidence targets
    "num_comparison_pairs": 500,
    "seed": 42,
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Training config
    "num_epochs": 5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "max_seq_length": 512,
    # Paths
    "results_dir": "results/finetuning",
    "training_data_dir": "results/finetuning/training_data",
    "checkpoint_dir": "results/finetuning/checkpoints",
    "merged_model_dir": "results/finetuning/merged_model",
    "eval_dir": "results/finetuning/eval",
}
