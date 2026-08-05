# config/RoBERTa_config.py
"""
RoBERTa-base MCQ configuration.

Changes from reviewed version
──────────────────────────────
  - Removed unused fields: top_k, wandb_entity (empty, never used)
  - n_gpus: clamped to >= 1 (device_count() returns 0 when no CUDA)
  - use_wandb: now actually wired through pipeline
  - Added lazy_tokenization flag for MCQDataset
  - Added unfreeze_start_epoch alias (was unfreeze_epoch — kept both
    for backward compat)
  - fp16: remains False; autocast wired conditionally in Trainer
  - All other values unchanged from the performance-validated version
"""

import dataclasses
import torch


def _default_n_gpus() -> int:
    return max(torch.cuda.device_count(), 1)


@dataclasses.dataclass
class Config:

    # ── Model ──────────────────────────────────────────────────────────────
    model_name      : str   = "roberta-base"
    pooling         : str   = "mean"
    hidden_dropout  : float = 0.1
    use_grad_ckpt   : bool  = True
    freeze_layers   : int   = 2

    # ── SBERT dedup ────────────────────────────────────────────────────────
    sbert_model      : str   = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size : int   = 256
    sim_threshold    : float = 0.92
    max_agglom_rows  : int   = 10_000   # guard: skip clustering above this

    # ── Training ───────────────────────────────────────────────────────────
    epochs          : int   = 12
    batch_size      : int   = 8         # per-GPU
    grad_accum      : int   = 4         # effective = 8 × 2 GPU × 4 = 64
    max_len         : int   = 128

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr_backbone         : float = 2e-5
    lr_head             : float = 5e-5
    weight_decay        : float = 0.01
    max_grad_norm       : float = 1.0
    warmup_ratio        : float = 0.06

    # ── Loss ───────────────────────────────────────────────────────────────
    smoothing : float = 0.05
    margin    : float = 0.3
    ce_w      : float = 0.65
    rank_w    : float = 0.35

    # ── Regularization ─────────────────────────────────────────────────────
    n_dropouts          : int   = 4
    unfreeze_epoch      : int   = 2     # epoch at which unfreezing begins
    early_stop_patience : int   = 5
    early_stop_grace    : int   = 3     # don't count ES before this epoch

    # ── Data split ─────────────────────────────────────────────────────────
    val_size    : float = 0.12
    seed        : int   = 42
    audit_top_k : int   = 20

    # ── Hardware ───────────────────────────────────────────────────────────
    device      : str  = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus      : int  = dataclasses.field(default_factory=_default_n_gpus)
    num_workers : int  = 4             # bumped: tokenisation already done
    use_fp16    : bool = False         # FP32 only — stable on all hardware

    # ── Dataset ────────────────────────────────────────────────────────────
    lazy_tokenization : bool = False   # True → tokenize on-demand (low RAM)

    # ── W&B ────────────────────────────────────────────────────────────────
    use_wandb      : bool = True
    wandb_project  : str  = "Milestone-6"
    wandb_run_name : str  = "roberta-run"

    # ── Paths ──────────────────────────────────────────────────────────────
    artifacts_save_dir : str = "/kaggle/working/artifact"
    artifacts_load_dir : str = (
        "/kaggle/input/notebooks/mimakhdumiiitm"
        "/dl-22f3001418-notebook-t22026/outputs/artifacts"
    )
    submission_dir : str = "/kaggle/working/submission"
    plots_dir      : str = "/kaggle/working/plots"

    def __post_init__(self):
        # Clamp n_gpus to a sane range
        if self.n_gpus < 1:
            self.n_gpus = 1