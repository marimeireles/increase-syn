#!/usr/bin/env python
"""
Phase F1: Generate consistency-based confidence targets.

Loads Gemma 3 4B-IT, generates multiple responses per question from
MMLU-Pro, GSM8K, and TriviaQA, computes consistency scores,
bins by consistency, and creates calibrated training data.
"""

import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging
from finetuning.config import FT_CONFIG
from finetuning.consistency_scorer import ConsistencyScorer
from finetuning.data_preparation import assign_confidence_targets, build_training_examples

logger = logging.getLogger(__name__)


def load_datasets(config):
    """Load and subsample datasets, returning unified question list."""
    from datasets import load_dataset

    questions_train = []
    questions_test = []

    # MMLU-Pro
    ds_cfg = config["datasets"]["mmlu_pro"]
    logger.info(f"Loading MMLU-Pro...")
    ds = load_dataset(ds_cfg["name"], split="test")  # MMLU-Pro only has test split
    ds = ds.shuffle(seed=config["seed"])

    total_needed = ds_cfg["train_size"] + ds_cfg["test_size"]
    for i, item in enumerate(ds):
        if i >= total_needed:
            break
        # Format as MCQ
        options = item.get("options", [])
        option_str = "\n".join(
            f"({chr(65+j)}) {opt}" for j, opt in enumerate(options)
        )
        q = {
            "question": f"{item['question']}\n{option_str}",
            "ground_truth": item["answer"],
            "domain": "mcq",
            "max_new_tokens": config["max_new_tokens_mcq"],
        }
        if i < ds_cfg["train_size"]:
            questions_train.append(q)
        else:
            questions_test.append(q)

    logger.info(f"MMLU-Pro: {len([q for q in questions_train if q['domain'] == 'mcq'])} train, "
                f"{len([q for q in questions_test if q['domain'] == 'mcq'])} test")

    # GSM8K
    ds_cfg = config["datasets"]["gsm8k"]
    logger.info(f"Loading GSM8K...")
    ds = load_dataset(ds_cfg["name"], ds_cfg["config"], split="train")
    ds = ds.shuffle(seed=config["seed"])

    total_needed = ds_cfg["train_size"] + ds_cfg["test_size"]
    for i, item in enumerate(ds):
        if i >= total_needed:
            break
        # Extract final answer from GSM8K format
        answer = item["answer"].split("####")[-1].strip()
        q = {
            "question": item["question"],
            "ground_truth": answer,
            "domain": "math",
            "max_new_tokens": config["max_new_tokens_math"],
        }
        if i < ds_cfg["train_size"]:
            questions_train.append(q)
        else:
            questions_test.append(q)

    logger.info(f"GSM8K: {len([q for q in questions_train if q['domain'] == 'math'])} train, "
                f"{len([q for q in questions_test if q['domain'] == 'math'])} test")

    # TriviaQA
    ds_cfg = config["datasets"]["trivia_qa"]
    logger.info(f"Loading TriviaQA...")
    ds = load_dataset(ds_cfg["name"], ds_cfg["config"], split="train")
    ds = ds.shuffle(seed=config["seed"])

    total_needed = ds_cfg["train_size"] + ds_cfg["test_size"]
    for i, item in enumerate(ds):
        if i >= total_needed:
            break
        # Use first alias as ground truth
        answer = item["answer"]["value"]
        q = {
            "question": item["question"],
            "ground_truth": answer,
            "domain": "trivia",
            "max_new_tokens": config["max_new_tokens_trivia"],
        }
        if i < ds_cfg["train_size"]:
            questions_train.append(q)
        else:
            questions_test.append(q)

    logger.info(f"TriviaQA: {len([q for q in questions_train if q['domain'] == 'trivia'])} train, "
                f"{len([q for q in questions_test if q['domain'] == 'trivia'])} test")

    logger.info(f"Total: {len(questions_train)} train, {len(questions_test)} test")
    return questions_train, questions_test


def main():
    setup_logging()
    config = FT_CONFIG

    # Check if training data already exists
    data_dir = config["training_data_dir"]
    train_path = os.path.join(data_dir, "training_examples.json")
    if os.path.exists(train_path):
        logger.info(f"Training data already exists at {train_path}. Skipping.")
        return

    set_seed(config["seed"])
    device = get_device()

    # Create output directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config["eval_dir"], exist_ok=True)

    # Load datasets
    questions_train, questions_test = load_datasets(config)

    # Save test questions for later evaluation
    test_path = os.path.join(config["eval_dir"], "test_questions.json")
    with open(test_path, "w") as f:
        json.dump(questions_test, f, indent=2)
    logger.info(f"Saved {len(questions_test)} test questions to {test_path}")

    # Load model
    logger.info("Loading Gemma 3 4B-IT for consistency scoring...")
    model, tokenizer = load_model_and_tokenizer(
        config["base_model"], device, torch_dtype="bfloat16",
    )

    # Score questions
    scorer = ConsistencyScorer(
        model, tokenizer, device,
        num_samples=config["num_samples"],
        temperature=config["temperature"],
    )

    logger.info("Scoring training questions for consistency...")
    scored = scorer.score_dataset(questions_train, progress_desc="Consistency scoring")

    # Save raw scores
    scores_path = os.path.join(data_dir, "consistency_scores.json")
    # Remove responses to save space (they can be very long)
    scores_to_save = [{k: v for k, v in s.items() if k != "responses"} for s in scored]
    with open(scores_path, "w") as f:
        json.dump(scores_to_save, f, indent=2)
    logger.info(f"Saved consistency scores to {scores_path}")

    # Assign confidence targets
    targeted = assign_confidence_targets(
        scored,
        noise_range=config["noise_range"],
        seed=config["seed"],
    )

    # Build training examples
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

    # Spot-check
    logger.info("\n=== Spot check (first 3 examples) ===")
    for i, ex in enumerate(training_examples[:3]):
        logger.info(f"\nExample {i+1} ({ex['type']}):")
        logger.info(ex["text"][:300] + "...")

    # Free model
    del model, scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Phase F1 complete.")


if __name__ == "__main__":
    main()
