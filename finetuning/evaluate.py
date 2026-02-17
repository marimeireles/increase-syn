"""
Calibration evaluation for base and fine-tuned models.

Metrics: ECE (Expected Calibration Error), AUC (discrimination).
"""

import logging
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_confidence(text: str) -> Optional[float]:
    """Extract a confidence score from model output using multiple regex patterns."""
    patterns = [
        r"confidence\s*(?:score\s*)?(?:is\s*)?(\d+\.?\d*)",
        r"confidence:\s*(\d+\.?\d*)",
        r"(\d\.\d+)\s*$",
        r"(\d\.\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val
            # If > 1 and <= 100, treat as percentage
            if 1.0 < val <= 100.0:
                return val / 100.0
    return None


def extract_answer_and_confidence(text: str, domain: str) -> Tuple[Optional[str], Optional[float]]:
    """Extract both answer and confidence from model output."""
    from finetuning.consistency_scorer import (
        extract_mcq_answer, extract_math_answer, extract_trivia_answer,
    )

    confidence = extract_confidence(text)

    if domain == "mcq":
        answer = extract_mcq_answer(text)
    elif domain == "math":
        answer = extract_math_answer(text)
    else:
        answer = extract_trivia_answer(text)

    return answer, confidence


@torch.no_grad()
def generate_with_confidence(model, tokenizer, question: str,
                             max_new_tokens: int, device) -> str:
    """Generate a response with confidence from the model."""
    from finetuning.prompt_templates import SYSTEM_PROMPT_CONFIDENCE

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONFIDENCE},
        {"role": "user", "content": question},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy for evaluation
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_calibration(model, tokenizer, test_questions: List[Dict],
                         device, model_name: str = "model") -> Dict:
    """
    Evaluate calibration on a test set.

    Args:
        model: the model to evaluate
        tokenizer: tokenizer
        test_questions: list of dicts with keys: question, ground_truth, domain, max_new_tokens
        device: torch device
        model_name: label for logging

    Returns:
        dict with keys: ece, auc, parse_rate, per_domain_ece, per_domain_auc,
                        predictions (list of dicts)
    """
    predictions = []

    for q in tqdm(test_questions, desc=f"Evaluating {model_name}"):
        response = generate_with_confidence(
            model, tokenizer, q["question"], q["max_new_tokens"], device,
        )
        answer, confidence = extract_answer_and_confidence(response, q["domain"])

        # Check correctness
        is_correct = False
        if answer is not None:
            gt = q["ground_truth"].strip().lower()
            if q["domain"] == "mcq":
                is_correct = answer.upper() == gt.upper()
            elif q["domain"] == "math":
                try:
                    is_correct = abs(float(answer) - float(gt)) < 1e-6
                except (ValueError, TypeError):
                    is_correct = answer == gt
            else:
                is_correct = (answer.lower().strip() == gt or
                              gt in answer.lower().strip())

        predictions.append({
            "question": q["question"],
            "domain": q["domain"],
            "ground_truth": q["ground_truth"],
            "predicted_answer": answer,
            "confidence": confidence if confidence is not None else 0.5,
            "confidence_parsed": confidence is not None,
            "is_correct": is_correct,
            "response": response,
        })

    # Compute metrics
    parsed = [p for p in predictions if p["confidence_parsed"]]
    parse_rate = len(parsed) / len(predictions) if predictions else 0.0

    ece = compute_ece(predictions)
    auc = compute_auc(predictions)

    # Per-domain metrics
    domains = set(p["domain"] for p in predictions)
    per_domain_ece = {}
    per_domain_auc = {}
    for d in domains:
        d_preds = [p for p in predictions if p["domain"] == d]
        per_domain_ece[d] = compute_ece(d_preds)
        per_domain_auc[d] = compute_auc(d_preds)

    logger.info(f"[{model_name}] ECE={ece:.4f}, AUC={auc:.4f}, "
                f"Parse rate={parse_rate:.2%}")
    for d in sorted(domains):
        logger.info(f"  {d}: ECE={per_domain_ece[d]:.4f}, AUC={per_domain_auc[d]:.4f}")

    return {
        "ece": ece,
        "auc": auc,
        "parse_rate": parse_rate,
        "per_domain_ece": per_domain_ece,
        "per_domain_auc": per_domain_auc,
        "predictions": predictions,
    }


def compute_ece(predictions: List[Dict], num_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = sum(|B_m|/N * |acc(B_m) - conf(B_m)|) for m in bins
    """
    if not predictions:
        return 0.0

    confidences = np.array([p["confidence"] for p in predictions])
    corrects = np.array([float(p["is_correct"]) for p in predictions])
    n = len(predictions)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_acc = corrects[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_auc(predictions: List[Dict]) -> float:
    """
    Compute AUC for discrimination: P(conf_correct > conf_incorrect).

    Uses the Wilcoxon-Mann-Whitney statistic.
    """
    correct_confs = [p["confidence"] for p in predictions if p["is_correct"]]
    incorrect_confs = [p["confidence"] for p in predictions if not p["is_correct"]]

    if not correct_confs or not incorrect_confs:
        return 0.5  # undefined, return chance

    # Count concordant pairs
    concordant = 0
    tied = 0
    total = len(correct_confs) * len(incorrect_confs)

    for cc in correct_confs:
        for ic in incorrect_confs:
            if cc > ic:
                concordant += 1
            elif cc == ic:
                tied += 1

    auc = (concordant + 0.5 * tied) / total
    return float(auc)
