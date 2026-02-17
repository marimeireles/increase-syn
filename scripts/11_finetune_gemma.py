#!/usr/bin/env python
"""
Phase F2: LoRA fine-tuning of Gemma 3 4B-IT for calibrated confidence.

Loads training data from Phase F1, applies LoRA to attention projections,
and trains with SFTTrainer.
"""

import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging
from finetuning.config import FT_CONFIG
from finetuning.train import run_finetuning

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = FT_CONFIG

    # Check if adapter already exists
    adapter_dir = os.path.join(config["checkpoint_dir"], "final_adapter")
    if os.path.exists(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info(f"LoRA adapter already exists at {adapter_dir}. Skipping.")
        return

    set_seed(config["seed"])
    device = get_device()

    # Load training data
    train_path = os.path.join(config["training_data_dir"], "training_examples.json")
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}. Run 10_prepare_finetune_data.py first.")
        sys.exit(1)

    with open(train_path) as f:
        training_examples = json.load(f)
    logger.info(f"Loaded {len(training_examples)} training examples")

    # Load model
    logger.info("Loading Gemma 3 4B-IT for fine-tuning...")
    model, tokenizer = load_model_and_tokenizer(
        config["base_model"], device, torch_dtype="bfloat16",
    )

    # Run fine-tuning
    trainer = run_finetuning(model, tokenizer, training_examples, config)

    # Log training loss curve
    if trainer.state.log_history:
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            logger.info(f"Training loss: start={losses[0]:.4f}, end={losses[-1]:.4f}")
            if len(losses) > 5:
                logger.info(f"Loss curve (sampled): {[f'{l:.4f}' for l in losses[::max(1, len(losses)//10)]]}")

    # Free model
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Phase F2 complete.")


if __name__ == "__main__":
    main()
