"""
LoRA fine-tuning for metacognitive confidence calibration.

Uses TRL SFTTrainer with LoRA on attention projections only.
"""

import logging
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def create_lora_config(config: dict) -> LoraConfig:
    """Create LoRA configuration from fine-tuning config."""
    return LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )


def create_training_args(config: dict, output_dir: str) -> TrainingArguments:
    """Create training arguments from fine-tuning config."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=config["warmup_ratio"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=config["seed"],
        dataloader_pin_memory=False,
    )


def run_finetuning(model, tokenizer, training_examples: list, config: dict):
    """
    Fine-tune model with LoRA using SFTTrainer.

    Args:
        model: base model (loaded in bfloat16)
        tokenizer: tokenizer
        training_examples: list of dicts with 'text' key
        config: FT_CONFIG dict

    Returns:
        trainer: the SFTTrainer after training (model has LoRA adapters)
    """
    output_dir = config["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Create LoRA config and apply
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")

    # Build HF Dataset
    dataset = Dataset.from_list(training_examples)
    logger.info(f"Training dataset: {len(dataset)} examples")

    # Training arguments
    training_args = create_training_args(config, output_dir)

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=config["max_seq_length"],
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save final adapter
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    logger.info(f"Saved LoRA adapter to {final_adapter_dir}")

    return trainer
