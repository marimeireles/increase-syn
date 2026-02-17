"""
Data preparation for metacognitive fine-tuning.

Converts consistency scores into calibrated confidence targets,
then formats training examples in Gemma chat template.
"""

import logging
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from finetuning.prompt_templates import (
    format_single_question_messages,
    format_comparison_messages,
)

logger = logging.getLogger(__name__)


def assign_confidence_targets(scored_questions: List[Dict],
                              num_bins: int = 10,
                              noise_range: float = 0.05,
                              max_bin_imbalance: float = 0.20,
                              seed: int = 42) -> List[Dict]:
    """
    Bin questions by consistency, compute empirical accuracy per bin,
    assign target confidence = accuracy(bin) + Uniform(-noise, +noise).

    Args:
        scored_questions: list of dicts from ConsistencyScorer.score_dataset()
        num_bins: number of consistency bins
        noise_range: epsilon for noise added to targets
        max_bin_imbalance: max frequency difference between top two bins
        seed: random seed

    Returns:
        list of dicts with added 'target_confidence' and 'bin_idx' keys
    """
    rng = np.random.RandomState(seed)

    # Filter to questions with valid answers
    valid = [q for q in scored_questions if q["modal_answer"] is not None]
    if not valid:
        logger.warning("No valid scored questions found!")
        return []

    logger.info(f"Assigning confidence targets to {len(valid)} questions")

    # Bin by consistency score
    consistencies = np.array([q["consistency"] for q in valid])
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(consistencies, bin_edges[1:], right=True)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute empirical accuracy per bin
    bin_accuracy = {}
    bin_counts = {}
    for b in range(num_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            bin_accuracy[b] = 0.5  # default for empty bins
            bin_counts[b] = 0
            continue
        correct = sum(1 for q, m in zip(valid, mask) if m and q["is_correct"])
        total = int(mask.sum())
        bin_accuracy[b] = correct / total
        bin_counts[b] = total

    logger.info("Bin accuracies:")
    for b in range(num_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        logger.info(f"  Bin {b} [{lo:.1f}-{hi:.1f}]: accuracy={bin_accuracy[b]:.3f}, "
                     f"count={bin_counts[b]}")

    # Subsample overrepresented bins
    counts_sorted = sorted(bin_counts.values(), reverse=True)
    if len(counts_sorted) >= 2 and counts_sorted[0] > 0:
        total_n = sum(bin_counts.values())
        freq_top = counts_sorted[0] / total_n
        freq_second = counts_sorted[1] / total_n if counts_sorted[1] > 0 else 0
        if freq_top - freq_second > max_bin_imbalance:
            max_per_bin = int(counts_sorted[1] * 1.2) if counts_sorted[1] > 0 else counts_sorted[0]
            logger.info(f"Subsampling overrepresented bins to max {max_per_bin} per bin")
            # Mark excess items for removal
            bin_item_indices = {b: [] for b in range(num_bins)}
            for idx, b in enumerate(bin_indices):
                bin_item_indices[b].append(idx)
            keep_mask = np.ones(len(valid), dtype=bool)
            for b, items in bin_item_indices.items():
                if len(items) > max_per_bin:
                    remove = rng.choice(items, size=len(items) - max_per_bin, replace=False)
                    keep_mask[remove] = False
            valid = [q for q, k in zip(valid, keep_mask) if k]
            bin_indices = bin_indices[keep_mask]
            logger.info(f"After subsampling: {len(valid)} questions")

    # Assign targets
    results = []
    for q, b in zip(valid, bin_indices):
        noise = rng.uniform(-noise_range, noise_range)
        target = np.clip(bin_accuracy[b] + noise, 0.0, 1.0)
        q_copy = dict(q)
        q_copy["target_confidence"] = float(target)
        q_copy["bin_idx"] = int(b)
        results.append(q_copy)

    return results


def build_training_examples(targeted_questions: List[Dict],
                            num_comparison_pairs: int = 500,
                            seed: int = 42,
                            tokenizer=None) -> List[Dict]:
    """
    Build training examples in chat format.

    Creates:
    1. Single-question confidence examples
    2. Pairwise comparison examples

    Args:
        targeted_questions: list from assign_confidence_targets()
        num_comparison_pairs: number of pairwise examples to generate
        seed: random seed
        tokenizer: HF tokenizer (needed for apply_chat_template)

    Returns:
        list of dicts with 'text' key (formatted chat string)
    """
    rng = random.Random(seed)
    examples = []

    # 1. Single-question examples
    for q in targeted_questions:
        answer = q["modal_answer"]
        confidence = q["target_confidence"]
        messages = format_single_question_messages(
            q["question"], answer, confidence
        )
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            # Fallback: simple concatenation
            text = "\n".join(
                f"<{m['role']}>{m['content']}" for m in messages
            )
        examples.append({"text": text, "type": "single"})

    # 2. Pairwise comparison examples
    if len(targeted_questions) >= 2:
        # Create pairs from different bins
        by_bin = {}
        for q in targeted_questions:
            b = q["bin_idx"]
            if b not in by_bin:
                by_bin[b] = []
            by_bin[b].append(q)

        bins_with_items = [b for b in sorted(by_bin.keys()) if len(by_bin[b]) > 0]
        pairs_created = 0

        for _ in range(num_comparison_pairs * 3):  # oversample, then truncate
            if pairs_created >= num_comparison_pairs:
                break
            if len(bins_with_items) < 2:
                # Fall back to random pairs
                q1, q2 = rng.sample(targeted_questions, 2)
            else:
                b1, b2 = rng.sample(bins_with_items, 2)
                q1 = rng.choice(by_bin[b1])
                q2 = rng.choice(by_bin[b2])

            # The question with higher target confidence is the "correct" choice
            if q1["target_confidence"] >= q2["target_confidence"]:
                choice = 1
            else:
                choice = 2

            messages = format_comparison_messages(
                q1["question"], q2["question"], choice
            )
            if tokenizer is not None:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            else:
                text = "\n".join(
                    f"<{m['role']}>{m['content']}" for m in messages
                )
            examples.append({"text": text, "type": "comparison"})
            pairs_created += 1

    logger.info(f"Built {len(examples)} training examples "
                f"({sum(1 for e in examples if e['type'] == 'single')} single, "
                f"{sum(1 for e in examples if e['type'] == 'comparison')} comparison)")

    # Shuffle
    rng.shuffle(examples)
    return examples
