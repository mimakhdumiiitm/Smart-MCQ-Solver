# config/DeBERTa_config.py
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # ── paths ─────────────────────────────────────────────────────────────────
    artifacts_save_dir : str  = "/kaggle/working/artifact"
    artifacts_load_dir : str  = "/kaggle/input/notebooks/mimakhdumiiitm/dl-22f3001418-notebook-t22026/outputs/artifacts"
    submission_dir     : str  = "/kaggle/working/submission"
    plots_dir          : str  = "/kaggle/working/plots"

    # ── model ──────────────────────────────────────────────────────────────
    model_name         : str  = "microsoft/deberta-v3-base"

    # ── dedup ──────────────────────────────────────────────────────────────
    sim_threshold      : float = 0.85
    bow_max_features   : int   = 30_000
    bow_ngram_max      : int   = 2
    val_size           : float = 0.15
    audit_top_k        : int   = 20

    # ── tokenizer ──────────────────────────────────────────────────────────
    max_len            : int   = 128   # reduced: 128 is enough for MCQ
                                       # 256 → 2× slower, no accuracy gain

    # ── training ───────────────────────────────────────────────────────────
    epochs             : int   = 8
    batch_size         : int   = 16    # increased: GPU has headroom with max_len=128
    grad_accum_steps   : int   = 2     # effective batch = 32
    lr                 : float = 2e-5  # slightly higher: better for small dataset
    weight_decay       : float = 0.01
    warmup_ratio       : float = 0.1
    max_grad_norm      : float = 1.0
    early_stop_patience: int   = 4

    # ── loss ───────────────────────────────────────────────────────────────
    smoothing          : float = 0.05
    margin             : float = 0.5
    ce_w               : float = 0.7
    rank_w             : float = 0.3

    # ── model regularization ───────────────────────────────────────────────
    classifier_dropout : float = 0.1
    freeze_layers      : int   = 0     # fine-tune all layers

    # ── hardware ───────────────────────────────────────────────────────────
    device             : str   = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus             : int   = torch.cuda.device_count()
    fp16               : bool  = True

    # ── misc ───────────────────────────────────────────────────────────────
    seed               : int   = 42
    max_vocab          : int   = 20_000
    min_freq           : int   = 2
    top_k              : int   = 3

    # ── W&B ────────────────────────────────────────────────────────────────
    use_wandb          : bool  = True
    wandb_project      : str   = "Smart-MCQ-Solver"
    wandb_entity       : str   = ""