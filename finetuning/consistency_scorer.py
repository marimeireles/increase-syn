"""
Consistency-based confidence target generation (Steyvers et al. method).

For each question, generate N responses at temperature=1 and compute the
fraction that match the modal answer (consistency score).
"""

import logging
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract a multiple-choice letter (A-J) from model output."""
    # Try common patterns
    patterns = [
        r"(?:the answer is|answer:)\s*\(?([A-Ja-j])\)?",
        r"^\s*\(?([A-Ja-j])\)?[\.\)\s]",
        r"\b([A-Ja-j])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract a numeric answer from math problem output."""
    # GSM8K style: look for #### pattern
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in text
    nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def extract_trivia_answer(text: str) -> Optional[str]:
    """Extract an open-ended answer (case-insensitive normalization)."""
    # Clean up: take first sentence/line as the answer
    text = text.strip()
    # Remove common prefixes
    for prefix in ["The answer is ", "Answer: ", "It's ", "It is "]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    # Take first line
    text = text.split("\n")[0].strip().rstrip(".")
    return text.lower().strip() if text else None


class ConsistencyScorer:
    """
    Generate multiple responses per question and compute consistency scores.
    """

    def __init__(self, model, tokenizer, device, num_samples=10, temperature=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.temperature = temperature

    @torch.no_grad()
    def _generate_responses(self, prompt: str, max_new_tokens: int,
                            num_samples: int) -> List[str]:
        """Generate multiple responses for a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            input_text, return_tensors="pt",
        ).input_ids.to(self.device)

        responses = []
        for _ in range(num_samples):
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = output[0, input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text)

        return responses

    def score_question(self, question: str, ground_truth: str,
                       domain: str, max_new_tokens: int) -> Dict:
        """
        Score a single question by generating num_samples responses.

        Args:
            question: the question text
            ground_truth: correct answer string
            domain: "mcq", "math", or "trivia"
            max_new_tokens: max tokens per response

        Returns:
            dict with keys: question, domain, ground_truth, consistency,
                            modal_answer, is_correct, num_valid, responses
        """
        responses = self._generate_responses(question, max_new_tokens, self.num_samples)

        # Extract answers
        extractor = {
            "mcq": extract_mcq_answer,
            "math": extract_math_answer,
            "trivia": extract_trivia_answer,
        }[domain]

        extracted = [extractor(r) for r in responses]
        valid = [a for a in extracted if a is not None]

        if not valid:
            return {
                "question": question,
                "domain": domain,
                "ground_truth": ground_truth,
                "consistency": 0.0,
                "modal_answer": None,
                "is_correct": False,
                "num_valid": 0,
                "responses": responses,
            }

        # Modal answer
        counts = Counter(valid)
        modal_answer, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(valid)

        # Check correctness
        gt_norm = ground_truth.strip().lower()
        if domain == "mcq":
            is_correct = modal_answer.upper() == gt_norm.upper()
        elif domain == "math":
            try:
                is_correct = abs(float(modal_answer) - float(gt_norm)) < 1e-6
            except (ValueError, TypeError):
                is_correct = modal_answer == gt_norm
        else:  # trivia
            is_correct = (modal_answer.lower().strip() == gt_norm or
                          gt_norm in modal_answer.lower().strip())

        return {
            "question": question,
            "domain": domain,
            "ground_truth": ground_truth,
            "consistency": consistency,
            "modal_answer": modal_answer,
            "is_correct": is_correct,
            "num_valid": len(valid),
            "responses": responses,
        }

    def score_dataset(self, questions: List[Dict],
                      progress_desc: str = "Scoring") -> List[Dict]:
        """
        Score a list of questions.

        Each question dict must have: question, ground_truth, domain, max_new_tokens
        """
        results = []
        for q in tqdm(questions, desc=progress_desc):
            result = self.score_question(
                question=q["question"],
                ground_truth=q["ground_truth"],
                domain=q["domain"],
                max_new_tokens=q["max_new_tokens"],
            )
            results.append(result)
        return results
