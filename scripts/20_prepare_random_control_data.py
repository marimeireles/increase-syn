#!/usr/bin/env python
"""
Prepare training data for random-confidence control experiment.

Reuses the EXISTING consistency scores from the metacog fine-tuning
(no need to re-run the expensive GPU scoring), but assigns random
confidence targets instead of calibrated ones.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, setup_logging
from finetuning.config_control import CTRL_FT_CONFIG
from finetuning.config import FT_CONFIG
from finetuning.data_preparation_control import assign_random_confidence_targets
from finetuning.data_preparation import build_training_examples

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = CTRL_FT_CONFIG

    # Check if training data already exists
    data_dir = config["training_data_dir"]
    train_path = os.path.join(data_dir, "training_examples.json")
    if os.path.exists(train_path):
        logger.info(f"Training data already exists at {train_path}. Skipping.")
        return

    set_seed(config["seed"])

    # Create output directories
    os.makedirs(data_dir, exist_ok=True)

    # Load EXISTING consistency scores from metacog fine-tuning
    metacog_scores_path = os.path.join(
        FT_CONFIG["training_data_dir"], "consistency_scores.json"
    )
    if not os.path.exists(metacog_scores_path):
        logger.error(
            f"Consistency scores not found at {metacog_scores_path}. "
            "Run 10_prepare_finetune_data.py first."
        )
        sys.exit(1)

    with open(metacog_scores_path) as f:
        scored_questions = json.load(f)
    logger.info(f"Loaded {len(scored_questions)} consistency scores from metacog data")

    # Assign RANDOM confidence targets (the key difference)
    targeted = assign_random_confidence_targets(
        scored_questions,
        seed=123,  # different from main seed to avoid correlation
    )

    # Load tokenizer for chat template formatting (no need for full model)
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer for chat template formatting...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    # Build training examples using the SAME format as metacog
    training_examples = build_training_examples(
        targeted,
        num_comparison_pairs=config["num_comparison_pairs"],
        seed=config["seed"],
        tokenizer=tokenizer,
    )

    # Save training examples
    with open(train_path, "w") as f:
        json.dump(training_examples, f, indent=2)
    logger.info(f"Saved {len(training_examples)} training examples to {train_path}")

    # Verify: same counts as metacog
    metacog_train_path = os.path.join(
        FT_CONFIG["training_data_dir"], "training_examples.json"
    )
    if os.path.exists(metacog_train_path):
        with open(metacog_train_path) as f:
            metacog_examples = json.load(f)
        n_metacog = len(metacog_examples)
        n_ctrl = len(training_examples)
        logger.info(f"Metacog examples: {n_metacog}, Control examples: {n_ctrl}")
        if n_metacog != n_ctrl:
            logger.warning(
                f"Count mismatch! Metacog={n_metacog}, Control={n_ctrl}. "
                "This may be due to different subsampling from random binning."
            )

    # Spot-check
    single_examples = [e for e in training_examples if e["type"] == "single"]
    comparison_examples = [e for e in training_examples if e["type"] == "comparison"]
    logger.info(f"Single examples: {len(single_examples)}, "
                f"Comparison examples: {len(comparison_examples)}")

    logger.info("\n=== Spot check (first 3 examples) ===")
    for i, ex in enumerate(training_examples[:3]):
        logger.info(f"\nExample {i+1} ({ex['type']}):")
        logger.info(ex["text"][:300] + "...")

    logger.info("Random control data preparation complete.")


if __name__ == "__main__":
    main()
