# config/RoBERTa_config.py
"""
Configuration for RoBERTa-base MCQ pipeline.
Tuned for T4 x2, MAP@3 > 0.788 target.
"""

import dataclasses
import torch
from typing import Optional


@dataclasses.dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name      : str   = "roberta-base"
    pooling         : str   = "weighted_layer"   # weighted_layer | mean | cls | attention
    hidden_dropout  : float = 0.1
    use_grad_ckpt   : bool  = True
    freeze_layers   : int   = 3                  # freeze bottom-N layers initially
    unfreeze_epoch  : int   = 2                  # start unfreezing from this epoch

    # ── SBERT (dedup & audit) ─────────────────────────────────────────────────
    sbert_model     : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size: int   = 256
    sim_threshold   : float = 0.85

    # ── Data ──────────────────────────────────────────────────────────────────
    max_len         : int   = 128
    val_size        : float = 0.12
    seed            : int   = 42
    num_workers     : int   = 2

    # ── Training ──────────────────────────────────────────────────────────────
    epochs          : int   = 12
    batch_size      : int   = 8                  # per-GPU; effective = 8×2×4=64
    grad_accum      : int   = 4
    use_fp16        : bool  = True               # BF16-safe via float32 scaler
    max_grad_norm   : float = 1.0

    # ── LR ────────────────────────────────────────────────────────────────────
    lr_backbone     : float = 1e-5
    lr_head         : float = 5e-5
    weight_decay    : float = 0.01
    warmup_ratio    : float = 0.10

    # ── Loss ──────────────────────────────────────────────────────────────────
    smoothing       : float = 0.05
    margin          : float = 0.3
    ce_w            : float = 0.65
    rank_w          : float = 0.35

    # ── Regularisation ────────────────────────────────────────────────────────
    multi_sample_dropout: bool  = True
    n_dropout_samples   : int   = 5
    dropout_low         : float = 0.10
    dropout_high        : float = 0.50

    # ── SWA ───────────────────────────────────────────────────────────────────
    use_swa         : bool  = True
    swa_start_epoch : int   = 7                  # start SWA from this epoch
    swa_lr          : float = 5e-6

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stop_patience : int = 5

    # ── Paths ──────────────────────────────────────────────────────────────
    artifacts_save_dir : str = "/kaggle/working/artifact"
    artifacts_load_dir : str = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/artifacts"
    submission_dir : str = "/kaggle/working/submission"
    plots_dir      : str = "/kaggle/working/plots"

        # ── W&B ────────────────────────────────────────────────────────────────
    use_wandb      : bool = True
    wandb_project  : str  = "Milestone-6"
    wandb_run_name : str  = "roberta-run"
    wandb_entity    : Optional[str] = None

    # ── Hardware ──────────────────────────────────────────────────────────────
    top_k   : int  = 3
    n_gpus  : int  = 2

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_top_k : int = 20

    # ── Device (set at runtime) ───────────────────────────────────────────────
    device  : str  = "cuda" if torch.cuda.is_available() else "cpu"