#!/usr/bin/env python
"""
Evaluate calibration of the random-confidence control model.

Compares base Gemma 3 4B-IT against the random-control fine-tuned version
on the SAME test set used for the metacog evaluation.
"""

import json
import logging
import os
import sys

import torch
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging
from finetuning.config_control import CTRL_FT_CONFIG
from finetuning.config import FT_CONFIG
from finetuning.evaluate import evaluate_calibration

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = CTRL_FT_CONFIG

    # Check if evaluation already done
    os.makedirs(config["eval_dir"], exist_ok=True)
    eval_path = os.path.join(config["eval_dir"], "calibration_results.json")
    if os.path.exists(eval_path):
        logger.info(f"Evaluation results already exist at {eval_path}. Skipping.")
        return

    set_seed(config["seed"])
    device = get_device()

    # Load test questions from the ORIGINAL metacog eval (same test set)
    test_path = os.path.join(FT_CONFIG["eval_dir"], "test_questions.json")
    if not os.path.exists(test_path):
        logger.error(
            f"Test questions not found at {test_path}. "
            "Run 10_prepare_finetune_data.py first."
        )
        sys.exit(1)

    with open(test_path) as f:
        test_questions = json.load(f)
    logger.info(f"Loaded {len(test_questions)} test questions")

    # --- Evaluate base model ---
    logger.info("=" * 60)
    logger.info("Evaluating BASE model (Gemma 3 4B-IT)")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        config["base_model"], device, torch_dtype="bfloat16",
    )

    base_results = evaluate_calibration(
        model, tokenizer, test_questions, device, model_name="base",
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Evaluate random-control fine-tuned model ---
    logger.info("=" * 60)
    logger.info("Evaluating RANDOM-CONTROL fine-tuned model (base + LoRA)")
    logger.info("=" * 60)

    adapter_dir = os.path.join(config["checkpoint_dir"], "final_adapter")
    if not os.path.exists(adapter_dir):
        logger.error(
            f"LoRA adapter not found at {adapter_dir}. "
            "Run 21_finetune_random_control.py first."
        )
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(
        config["base_model"], device, torch_dtype="bfloat16",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    ctrl_results = evaluate_calibration(
        model, tokenizer, test_questions, device, model_name="random_control",
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Compare results ---
    logger.info("=" * 60)
    logger.info("COMPARISON: Base vs Random-Control")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'Base':>10} {'Random-Ctrl':>12} {'Delta':>10}")
    logger.info("-" * 55)
    logger.info(
        f"{'ECE':<20} {base_results['ece']:>10.4f} {ctrl_results['ece']:>12.4f} "
        f"{ctrl_results['ece'] - base_results['ece']:>10.4f}"
    )
    logger.info(
        f"{'AUC':<20} {base_results['auc']:>10.4f} {ctrl_results['auc']:>12.4f} "
        f"{ctrl_results['auc'] - base_results['auc']:>10.4f}"
    )
    logger.info(
        f"{'Parse Rate':<20} {base_results['parse_rate']:>10.2%} "
        f"{ctrl_results['parse_rate']:>12.2%} "
        f"{ctrl_results['parse_rate'] - base_results['parse_rate']:>10.2%}"
    )

    # Save results
    results_summary = {
        "base": {
            "ece": base_results["ece"],
            "auc": base_results["auc"],
            "parse_rate": base_results["parse_rate"],
            "per_domain_ece": base_results["per_domain_ece"],
            "per_domain_auc": base_results["per_domain_auc"],
        },
        "random_control": {
            "ece": ctrl_results["ece"],
            "auc": ctrl_results["auc"],
            "parse_rate": ctrl_results["parse_rate"],
            "per_domain_ece": ctrl_results["per_domain_ece"],
            "per_domain_auc": ctrl_results["per_domain_auc"],
        },
    }

    with open(eval_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved calibration results to {eval_path}")

    # Save full predictions
    for name, results in [("base", base_results), ("random_control", ctrl_results)]:
        preds_path = os.path.join(config["eval_dir"], f"{name}_predictions.json")
        preds_clean = [
            {k: v for k, v in p.items() if k != "response"}
            for p in results["predictions"]
        ]
        with open(preds_path, "w") as f:
            json.dump(preds_clean, f, indent=2)

    logger.info("Random control evaluation complete.")


if __name__ == "__main__":
    main()
