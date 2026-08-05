# config/RoBERTa_config.py
"""
Configuration for RoBERTa-base MCQ pipeline.
Memory-optimised for T4 x2 (15 GiB each), targeting MAP@3 > 0.788.
"""

import dataclasses
import torch


@dataclasses.dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name      : str   = "roberta-base"
    pooling         : str   = "mean"          # mean is most memory-efficient
    hidden_dropout  : float = 0.1
    use_grad_ckpt   : bool  = True
    freeze_layers   : int   = 6               # freeze more → less memory in backward

    # ── SBERT dedup ────────────────────────────────────────────────────────
    sbert_model     : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size: int   = 256
    sim_threshold   : float = 0.85

    # ── Training ───────────────────────────────────────────────────────────
    epochs          : int   = 10
    batch_size      : int   = 4               # per-GPU; small to fit OOM
    grad_accum      : int   = 8               # effective = 4×2GPU×8 = 64
    max_len         : int   = 96              # reduced from 128 → big memory saving

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr_backbone     : float = 1e-5
    lr_head         : float = 8e-5
    weight_decay    : float = 0.01
    max_grad_norm   : float = 1.0
    warmup_ratio    : float = 0.1

    # ── Loss ───────────────────────────────────────────────────────────────
    smoothing       : float = 0.05
    margin          : float = 0.3
    ce_w            : float = 0.65
    rank_w          : float = 0.35
    rdrop_w         : float = 0.0             # DISABLED — halves backward memory

    # ── Regularization ─────────────────────────────────────────────────────
    n_dropouts      : int   = 3               # reduced from 5 → less memory
    unfreeze_epoch  : int   = 3
    early_stop_patience: int = 4

    # ── Data split ─────────────────────────────────────────────────────────
    val_size        : float = 0.1
    seed            : int   = 42
    audit_top_k     : int   = 20

    # ── Hardware ───────────────────────────────────────────────────────────
    device          : str   = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus          : int   = torch.cuda.device_count()
    num_workers     : int   = 2
    use_fp16        : bool  = False           # keep FP32 — no grad dtype errors

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_entity    : str   = ""
    top_k           : int   = 3
    use_wandb      : bool          = True
    wandb_project  : str           = 'Milestone-6'
    wandb_run_name : str           = 'roberta-run'

    # ── Paths ──────────────────────────────────────────────────────────────
    artifacts_save_dir : str  = "/kaggle/working/artifact"
    artifacts_load_dir : str  = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/artifacts"
    submission_dir     : str  = "/kaggle/working/submission"
    plots_dir          : str  = "/kaggle/working/plots"
