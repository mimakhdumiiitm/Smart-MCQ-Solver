# config/RoBERTa_config.py
"""
RoBERTa-base MCQ configuration.

Key fixes vs previous version
──────────────────────────────
  sim_threshold : 0.85 → 0.92   less aggressive dedup, keeps more data
  freeze_layers : 6    → 2      only freeze embeddings + bottom 2 layers
  max_len       : 96   → 128    restore full context
  batch_size    : 4    → 8      more stable gradient estimates
  grad_accum    : 8    → 4      effective batch = 8×2GPU×4 = 64 (same)
  lr_backbone   : 1e-5 → 2e-5   faster backbone adaptation on small data
  lr_head       : 8e-5 → 5e-5   balanced with backbone lr
  unfreeze_epoch: 3    → 2      start unfreezing earlier
  epochs        : 10   → 12     more training with early stopping guard
  warmup_ratio  : 0.1  → 0.06   shorter warmup → more cosine decay time
  pooling       : mean → mean    keep (memory safe, works well)
  n_dropouts    : 3    → 4      slight ensemble boost
"""

import dataclasses
import torch


@dataclasses.dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name      : str   = "roberta-base"
    pooling         : str   = "mean"
    hidden_dropout  : float = 0.1
    use_grad_ckpt   : bool  = True
    freeze_layers   : int   = 2         # only freeze bottom 2 + embeddings

    # ── SBERT dedup ────────────────────────────────────────────────────────
    sbert_model     : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size: int   = 256
    sim_threshold   : float = 0.92      # less aggressive → keep more data

    # ── Training ───────────────────────────────────────────────────────────
    epochs              : int   = 12
    batch_size          : int   = 8     # per-GPU
    grad_accum          : int   = 4     # effective = 8×2GPU×4 = 64
    max_len             : int   = 128   # restore full context

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr_backbone         : float = 2e-5  # higher → faster adaptation
    lr_head             : float = 5e-5
    weight_decay        : float = 0.01
    max_grad_norm       : float = 1.0
    warmup_ratio        : float = 0.06  # shorter warmup

    # ── Loss ───────────────────────────────────────────────────────────────
    smoothing           : float = 0.05
    margin              : float = 0.3
    ce_w                : float = 0.65
    rank_w              : float = 0.35

    # ── Regularization ─────────────────────────────────────────────────────
    n_dropouts          : int   = 4
    unfreeze_epoch      : int   = 2     # start unfreezing from epoch 2
    early_stop_patience : int   = 5     # more patience for small val set

    # ── Data split ─────────────────────────────────────────────────────────
    val_size            : float = 0.12  # slightly larger val for stable MAP@3
    seed                : int   = 42
    audit_top_k         : int   = 20

    # ── Hardware ───────────────────────────────────────────────────────────
    device      : str  = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus      : int  = torch.cuda.device_count()
    num_workers : int  = 2
    use_fp16    : bool = False          # FP32 only — no grad dtype errors

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
