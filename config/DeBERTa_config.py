# config/DeBERTa_config.py
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Config:
    # ── Dedup / Audit (SBERT) ─────────────────────────────────────────────────
    sbert_model       : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size  : int   = 256          # large batch → fast encoding
    sim_threshold     : float = 0.85         # cosine threshold for near-dupes
    audit_top_k       : int   = 20

    # ── MCQ Model (DeBERTa) ───────────────────────────────────────────────────
    model_name        : str   = "microsoft/deberta-v3-small"
    max_len           : int   = 128          # tokens per (Q, option) pair
    pooling           : str   = "mean"       # "cls" | "mean" | "attention"
    hidden_dropout    : float = 0.1
    freeze_layers     : int   = 6            # freeze bottom-N layers at start
    unfreeze_epoch    : int   = 3            # begin unfreezing from this epoch

    # ── Training ──────────────────────────────────────────────────────────────
    epochs            : int   = 12
    batch_size        : int   = 16
    grad_accum        : int   = 2            # effective batch = 32
    lr_backbone       : float = 8e-6
    lr_head           : float = 1e-4
    weight_decay      : float = 0.01
    max_grad_norm     : float = 1.0
    warmup_ratio      : float = 0.10
    early_stop_patience: int  = 5

    # ── Loss ──────────────────────────────────────────────────────────────────
    smoothing         : float = 0.05
    margin            : float = 0.3
    ce_w              : float = 0.7
    rank_w            : float = 0.3

    # ── Split ─────────────────────────────────────────────────────────────────
    val_size          : float = 0.10
    seed              : int   = 42
    top_k             : int   = 3

    # ── paths ─────────────────────────────────────────────────────────────────
    artifacts_save_dir : str  = "/kaggle/working/artifact"
    artifacts_load_dir : str  = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/artifacts"
    submission_dir     : str  = "/kaggle/working/submission"
    plots_dir          : str  = "/kaggle/working/plots"

    # ── Hardware ──────────────────────────────────────────────────────────────
    device            : str   = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus            : int   = torch.cuda.device_count()
    use_fp16          : bool  = True
    use_grad_ckpt     : bool  = True
    num_workers       : int   = 2

    # ── W&B ────────────────────────────────────────────────────────────────
    use_wandb      : bool          = True
    wandb_project  : str           = 'Milestone-6'
    wandb_entity   : Optional[str] = None
    wandb_run_name : str           = 'deberta-run'