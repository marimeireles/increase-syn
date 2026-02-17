#!/usr/bin/env python
"""
Phase F5: Run synergy pipeline on the fine-tuned merged model.

Thin wrapper that calls run_pipeline.py with --model gemma3-4b-it-ft.
"""

import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging
from finetuning.config import FT_CONFIG

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    merged_dir = FT_CONFIG["merged_model_dir"]
    if not os.path.exists(os.path.join(merged_dir, "config.json")):
        logger.error(f"Merged model not found at {merged_dir}. Run 13_merge_lora.py first.")
        sys.exit(1)

    # Run the standard pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(script_dir, "run_pipeline.py")

    cmd = [
        sys.executable, pipeline_script,
        "--model", "gemma3-4b-it-ft",
        "--phases", "1", "2", "3", "4", "6",
        "--max-workers", "32",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Pipeline failed with return code {result.returncode}")
        sys.exit(result.returncode)

    logger.info("Phase F5 complete.")


if __name__ == "__main__":
    main()
