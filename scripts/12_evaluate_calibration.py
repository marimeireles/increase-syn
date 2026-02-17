#!/usr/bin/env python
"""
Phase F3: Evaluate calibration of base and fine-tuned models.

Compares ECE and AUC between the base Gemma 3 4B-IT and the LoRA fine-tuned version.
"""

import json
import logging
import os
import sys

import torch
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging
from finetuning.config import FT_CONFIG
from finetuning.evaluate import evaluate_calibration

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = FT_CONFIG

    # Check if evaluation already done
    eval_path = os.path.join(config["eval_dir"], "calibration_results.json")
    if os.path.exists(eval_path):
        logger.info(f"Evaluation results already exist at {eval_path}. Skipping.")
        return

    set_seed(config["seed"])
    device = get_device()

    # Load test questions
    test_path = os.path.join(config["eval_dir"], "test_questions.json")
    if not os.path.exists(test_path):
        logger.error(f"Test questions not found at {test_path}. Run 10_prepare_finetune_data.py first.")
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

    # Free base model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Evaluate fine-tuned model (base + LoRA adapter) ---
    logger.info("=" * 60)
    logger.info("Evaluating FINE-TUNED model (base + LoRA)")
    logger.info("=" * 60)

    adapter_dir = os.path.join(config["checkpoint_dir"], "final_adapter")
    if not os.path.exists(adapter_dir):
        logger.error(f"LoRA adapter not found at {adapter_dir}. Run 11_finetune_gemma.py first.")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(
        config["base_model"], device, torch_dtype="bfloat16",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    ft_results = evaluate_calibration(
        model, tokenizer, test_questions, device, model_name="finetuned",
    )

    # Free model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Compare results ---
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    logger.info("-" * 55)
    logger.info(f"{'ECE':<20} {base_results['ece']:>10.4f} {ft_results['ece']:>12.4f} "
                f"{ft_results['ece'] - base_results['ece']:>10.4f}")
    logger.info(f"{'AUC':<20} {base_results['auc']:>10.4f} {ft_results['auc']:>12.4f} "
                f"{ft_results['auc'] - base_results['auc']:>10.4f}")
    logger.info(f"{'Parse Rate':<20} {base_results['parse_rate']:>10.2%} {ft_results['parse_rate']:>12.2%} "
                f"{ft_results['parse_rate'] - base_results['parse_rate']:>10.2%}")

    # Save results (without raw predictions to keep file small)
    results_summary = {
        "base": {
            "ece": base_results["ece"],
            "auc": base_results["auc"],
            "parse_rate": base_results["parse_rate"],
            "per_domain_ece": base_results["per_domain_ece"],
            "per_domain_auc": base_results["per_domain_auc"],
        },
        "finetuned": {
            "ece": ft_results["ece"],
            "auc": ft_results["auc"],
            "parse_rate": ft_results["parse_rate"],
            "per_domain_ece": ft_results["per_domain_ece"],
            "per_domain_auc": ft_results["per_domain_auc"],
        },
    }

    os.makedirs(config["eval_dir"], exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved calibration results to {eval_path}")

    # Save full predictions separately
    for name, results in [("base", base_results), ("finetuned", ft_results)]:
        preds_path = os.path.join(config["eval_dir"], f"{name}_predictions.json")
        # Remove response text to save space
        preds_clean = [
            {k: v for k, v in p.items() if k != "response"}
            for p in results["predictions"]
        ]
        with open(preds_path, "w") as f:
            json.dump(preds_clean, f, indent=2)

    logger.info("Phase F3 complete.")


if __name__ == "__main__":
    main()
