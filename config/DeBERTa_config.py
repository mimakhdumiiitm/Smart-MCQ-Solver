# config/DeBERTa_config.py
"""
DeBERTa-v3 configuration — mirrors BiLSTM_config structure exactly.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── paths ─────────────────────────────────────────────────────────────────
    artifacts_save_dir : str  = "/kaggle/working/artifact"
    artifacts_load_dir : str  = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/artifacts"
    submission_dir     : str  = "/kaggle/working/submission"
    plots_dir          : str  = "/kaggle/working/plots"

    # ── pretrained model ──────────────────────────────────────────────────────
    model_name         : str  = "microsoft/deberta-v3-base"
    #   alternatives:
    #     "microsoft/deberta-v3-small"   → faster, less accurate
    #     "microsoft/deberta-v3-large"   → slower, more accurate

    # ── data / dedup ──────────────────────────────────────────────────────────
    sim_threshold      : float = 0.85
    bow_max_features   : int   = 30_000
    bow_ngram_max      : int   = 2
    val_size           : float = 0.15
    audit_top_k        : int   = 20

    # ── tokenizer ─────────────────────────────────────────────────────────────
    max_len            : int   = 256   # tokens per (question + option) pair
    #   DeBERTa handles up to 512; 256 covers most MCQ examples

    # ── training ──────────────────────────────────────────────────────────────
    epochs             : int   = 10
    batch_size         : int   = 8     # per-GPU; gradient accumulation below
    grad_accum_steps   : int   = 4     # effective batch = 8 × 4 = 32
    lr                 : float = 1e-5  # lower LR than BiLSTM — fine-tuning
    weight_decay       : float = 0.01
    warmup_ratio       : float = 0.06  # fraction of total steps for warmup
    max_grad_norm      : float = 1.0
    early_stop_patience: int   = 4     # fewer epochs → smaller patience

    # ── loss ──────────────────────────────────────────────────────────────────
    smoothing          : float = 0.05  # lighter smoothing for pretrained model
    margin             : float = 0.5
    ce_w               : float = 0.7
    rank_w             : float = 0.3

    # ── scheduler ─────────────────────────────────────────────────────────────
    sched_patience     : int   = 2
    sched_factor       : float = 0.5

    # ── regularization ────────────────────────────────────────────────────────
    classifier_dropout : float = 0.1
    freeze_layers      : int   = 0     # freeze first N transformer layers (0 = none)

    # ── hardware ──────────────────────────────────────────────────────────────
    device             : str   = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus             : int   = torch.cuda.device_count()
    fp16               : bool  = True  # mixed precision — saves memory

    # ── misc ──────────────────────────────────────────────────────────────────
    seed               : int   = 42
    max_vocab          : int   = 20_000   # kept for interface parity (unused)
    min_freq           : int   = 2        # kept for interface parity (unused)
    top_k              : int   = 3

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb      : bool          = True
    wandb_project  : str           = 'Milestone-6'
    wandb_entity   : Optional[str] = None
    wandb_run_name : str           = 'DeROBERTa-run'