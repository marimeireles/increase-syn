#!/usr/bin/env python
"""
Phase F4: Merge LoRA adapter into base model.

Creates a standalone merged model that can be used by the synergy pipeline
without needing PEFT.
"""

import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging
from src.model_registry import detect_model_spec
from finetuning.config import FT_CONFIG
from finetuning.merge_adapter import merge_and_save

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = FT_CONFIG

    merged_dir = config["merged_model_dir"]

    # Check if already merged
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        logger.info(f"Merged model already exists at {merged_dir}. Verifying...")
        model, tokenizer = load_model_and_tokenizer(merged_dir, torch_dtype="bfloat16")
        spec = detect_model_spec(model)
        logger.info(f"Merged model: {spec.num_layers} layers, {spec.num_heads} heads/layer, "
                     f"head_dim={spec.head_dim}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Verification passed. Skipping merge.")
        return

    set_seed(config["seed"])
    device = get_device()

    # Check adapter exists
    adapter_dir = os.path.join(config["checkpoint_dir"], "final_adapter")
    if not os.path.exists(adapter_dir):
        logger.error(f"LoRA adapter not found at {adapter_dir}. Run 11_finetune_gemma.py first.")
        sys.exit(1)

    # Load base model
    logger.info("Loading base model for merging...")
    model, tokenizer = load_model_and_tokenizer(
        config["base_model"], device, torch_dtype="bfloat16",
    )

    # Verify base model architecture before merge
    spec_before = detect_model_spec(model)
    logger.info(f"Base model: {spec_before.num_layers} layers, {spec_before.num_heads} heads/layer")

    # Merge
    merged_model = merge_and_save(model, tokenizer, adapter_dir, merged_dir)

    # Verify merged model architecture
    spec_after = detect_model_spec(merged_model)
    logger.info(f"Merged model: {spec_after.num_layers} layers, {spec_after.num_heads} heads/layer")

    assert spec_before.num_layers == spec_after.num_layers, "Layer count changed after merge!"
    assert spec_before.num_heads == spec_after.num_heads, "Head count changed after merge!"
    assert spec_before.head_dim == spec_after.head_dim, "Head dim changed after merge!"

    # Quick generation test
    logger.info("Quick generation test...")
    test_prompt = "What is 2 + 2?"
    messages = [{"role": "user", "content": test_prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = merged_model.generate(
            input_ids, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    logger.info(f"Test response: {response[:200]}")

    # Free
    del merged_model, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Phase F4 complete.")


if __name__ == "__main__":
    main()
