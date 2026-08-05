# config/RoBERTa_config.py
"""
Configuration for RoBERTa-base MCQ pipeline.
Tuned for T4 x2, targeting MAP@3 > 0.788.
"""

import dataclasses
import torch


@dataclasses.dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name      : str   = "roberta-base"          # fixed backbone
    pooling         : str   = "mean"                  # mean | cls | attention | weighted
    hidden_dropout  : float = 0.1
    use_grad_ckpt   : bool  = True
    freeze_layers   : int   = 3                       # freeze bottom N layers initially

    # ── SBERT dedup ────────────────────────────────────────────────────────
    sbert_model     : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size: int   = 256
    sim_threshold   : float = 0.85

    # ── Training ───────────────────────────────────────────────────────────
    epochs          : int   = 12
    batch_size      : int   = 8                       # per GPU; effective = 8×2×4=64
    grad_accum      : int   = 4
    max_len         : int   = 128

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr_backbone     : float = 1e-5                    # slightly higher than DeBERTa
    lr_head         : float = 1e-4
    weight_decay    : float = 0.01
    max_grad_norm   : float = 1.0
    warmup_ratio    : float = 0.1

    # ── Loss ───────────────────────────────────────────────────────────────
    smoothing       : float = 0.05
    margin          : float = 0.3
    ce_w            : float = 0.6
    rank_w          : float = 0.3
    rdrop_w         : float = 0.1                     # R-Drop KL consistency weight

    # ── Regularization ─────────────────────────────────────────────────────
    n_dropouts      : int   = 5                       # multi-sample dropout count
    unfreeze_epoch  : int   = 2                       # start progressive unfreezing

    # ── Data split ─────────────────────────────────────────────────────────
    val_size        : float = 0.1
    seed            : int   = 42
    audit_top_k     : int   = 20

    # ── Hardware ───────────────────────────────────────────────────────────
    device          : str   = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus          : int   = torch.cuda.device_count()
    num_workers     : int   = 2
    use_fp16        : bool  = False                  

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