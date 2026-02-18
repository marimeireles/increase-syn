"""
Data preparation for random-confidence control fine-tuning.

Assigns uniform random confidence targets instead of calibrated ones,
keeping everything else identical (same questions, same answers, same format).
"""

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)


def assign_random_confidence_targets(scored_questions: List[Dict],
                                     num_bins: int = 10,
                                     seed: int = 123) -> List[Dict]:
    """
    Assign RANDOM confidence targets (uniform [0, 1]) instead of calibrated ones.

    This is the control condition: same questions, same answers, same output
    format, but the confidence values carry no meaningful signal.

    Uses seed=123 (different from main seed=42) to ensure random confidences
    don't accidentally correlate with any data ordering.

    Args:
        scored_questions: list of dicts from ConsistencyScorer.score_dataset()
        num_bins: number of bins (used only for bin_idx assignment for
                  compatibility with build_training_examples)
        seed: random seed for reproducibility

    Returns:
        list of dicts with added 'target_confidence' and 'bin_idx' keys
    """
    rng = np.random.RandomState(seed)

    # Filter to questions with valid answers (same as calibrated version)
    valid = [q for q in scored_questions if q["modal_answer"] is not None]
    if not valid:
        logger.warning("No valid scored questions found!")
        return []

    logger.info(f"Assigning RANDOM confidence targets to {len(valid)} questions")

    # Assign random confidence targets
    bin_edges = np.linspace(0, 1, num_bins + 1)
    results = []
    for q in valid:
        target = float(rng.uniform(0.0, 1.0))
        # Assign bin_idx based on the random confidence value
        # (needed by build_training_examples for comparison pair generation)
        bin_idx = int(np.digitize(target, bin_edges[1:], right=True))
        bin_idx = min(bin_idx, num_bins - 1)

        q_copy = dict(q)
        q_copy["target_confidence"] = target
        q_copy["bin_idx"] = bin_idx
        results.append(q_copy)

    # Log distribution summary
    targets = [r["target_confidence"] for r in results]
    logger.info(f"Random confidence stats: "
                f"mean={np.mean(targets):.3f}, std={np.std(targets):.3f}, "
                f"min={np.min(targets):.3f}, max={np.max(targets):.3f}")

    return results
