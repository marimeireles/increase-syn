Metacognitive Fine-tuning and Synergistic Core Analysis

  This project investigates whether improving a language model's metacognitive abilities changes the information-theoretic structure of its
  attention heads.

  Background

  Recent work (arxiv 2601.06851) identified a "synergistic core" in large language models — a set of attention heads in middle layers that
  exhibit high information-theoretic synergy, as measured by Integrated Information Decomposition (PhiID). These heads are disproportionately
  important for model behavior: ablating them causes far greater output divergence than removing other heads.

  Separately, Steyvers et al. (arxiv 2510.05126) showed that language models can be fine-tuned to produce well-calibrated confidence estimates
   through a consistency-based supervised learning approach, effectively improving the model's metacognition.

  Research Question

  Does metacognitive fine-tuning alter the synergistic core?

  If calibrated confidence estimation requires more integrated processing across attention heads, we would expect synergy to increase in
  middle layers after fine-tuning. If metacognition is implemented through parameter changes that don't affect inter-head information
  structure, the synergy profile should remain unchanged. Either outcome is informative.

  Approach

  1. Fine-tune Gemma 3 4B-IT for metacognition using the Steyvers et al. method — sampling the model multiple times per question, computing
  answer consistency, and training it to output calibrated confidence scores via LoRA
  2. Measure synergy in the fine-tuned model using the existing PhiID pipeline (pairwise PhiID across all attention heads, syn-red ranking,
  iterative ablation)
  3. Compare the fine-tuned model's synergy profile against the existing baseline Gemma 3 4B-IT results to identify whether and where the
  synergistic core shifted

  Key Details

  - Model: google/gemma-3-4b-it (8 heads, 34 layers, 272 total attention heads)
  - Fine-tuning: LoRA on attention projections only (q/k/v/o), trained on MMLU-Pro, GSM8K, and TriviaQA with consistency-derived confidence
  targets
  - Synergy measurement: PhiID with Gaussian estimator, all C(272,2) = 36,856 head pairs, 60 cognitive task prompts
  - Evaluation: ECE and AUC for calibration quality; paired statistical tests and effect sizes for synergy profile comparison
