"""
Merge LoRA adapter weights into the base model.

After merging, the model can be used directly without PEFT,
which is required for the synergy pipeline hooks to work correctly.
"""

import logging
import os

import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


def merge_and_save(base_model, tokenizer, adapter_path: str, output_dir: str):
    """
    Load LoRA adapter, merge into base model, save merged model.

    Args:
        base_model: the base model (same as used for fine-tuning)
        tokenizer: tokenizer
        adapter_path: path to saved LoRA adapter
        output_dir: where to save the merged model

    Returns:
        merged_model: the merged model (without PEFT wrapper)
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Verify: check parameter count matches base
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Merged model parameters: {num_params:,}")

    return model
