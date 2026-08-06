# config/RoBERTa_config.py
"""
RoBERTa-base MCQ config — memory-safe, fast, MAP@3 > 0.80 target.
T4 x2, torch 2.10, CUDA 12.8
"""

import dataclasses
import torch
from typing import Optional


@dataclasses.dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name      : str   = "roberta-base"
    pooling         : str   = "mean"          # mean is fast & low-memory
    hidden_dropout  : float = 0.1
    use_grad_ckpt   : bool  = True            # critical for T4 memory
    freeze_layers   : int   = 2               # freeze bottom 6/12 layers
    unfreeze_epoch  : int   = 2           #7     # unfreeze 1 layer/epoch from ep2

    # ── SBERT ─────────────────────────────────────────────────────────────────
    sbert_model      : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size : int   = 256
    sim_threshold    : float = 0.98

    # ── Data ──────────────────────────────────────────────────────────────────
    max_len          : int   = 96             # 96 >> 128: 25% less memory
    val_size         : float = 0.20
    seed             : int   = 42
    num_workers      : int   = 2

    # ── Training ──────────────────────────────────────────────────────────────
    epochs           : int   = 10 # 6
    batch_size       : int   = 4             # per step; eff=4×8=32
    grad_accum       : int   = 8
    use_fp16         : bool  = True
    max_grad_norm    : float = 1.0

    # ── LR ────────────────────────────────────────────────────────────────────
    lr_backbone      : float = 1e-5      # slightly higher → faster convergence
    lr_head          : float = 5e-5
    weight_decay     : float = 0.01
    warmup_ratio     : float = 0.12

    # ── Loss ──────────────────────────────────────────────────────────────────
    smoothing        : float = 0.05
    margin           : float = 0.3
    ce_w             : float = 0.80
    rank_w           : float = 0.20

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stop_patience : int = 4

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
    top_k    : int  = 3
    n_gpus   : int  = 2
    device   : str  = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_top_k : int = 20