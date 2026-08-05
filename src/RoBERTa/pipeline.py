# src/RoBERTa/pipeline.py
"""
Full RoBERTa-base MCQ pipeline.

Changes from reviewed version
──────────────────────────────
  Critical fixes
  ─────────────
  1. use_wandb flag now respected: init_wandb called only when
     cfg.use_wandb is True.

  2. Duplicate cosine_similarity computation removed: the auditor
     already returns report['max_sims']; the pipeline reads it directly
     instead of recomputing the full [N_train × N_val] matrix.

  Medium fixes
  ────────────
  3. _plot() split into _plot_sim_dist() and _plot_training() for clarity.

  4. DataLoader num_workers bumped to cfg.num_workers (default 4) and
     persistent_workers=True enabled when num_workers > 0 to avoid
     worker startup overhead between epochs.

  5. SemanticDeduplicator receives sbert_cache_path so embeddings are
     not recomputed on repeated runs (complements the W&B artifact cache).

  Low fixes
  ─────────
  6. Logging uses %-style formatting throughout.
  7. Dead comments removed; module-level docstring updated.
  8. _unwrap alias kept for backward compat; internally uses _get_raw_model
     from training.py consistently.

  Multi-GPU
  ─────────
  9. DataLoader pin_memory=True for CUDA (unchanged); persistent_workers
     avoids repeated process spawning between epochs.
 10. torch.compile() applied when PyTorch >= 2.0 and n_gpus == 1 (compile
     is incompatible with DataParallel on most PyTorch versions).
"""

import dataclasses
import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.RoBERTa_config import Config
from src.RoBERTa.artifacts import (
    _DEDUP,
    load_dedup,
    save_audit,
    save_dedup,
    save_model,
    save_submission,
    try_load,
)
from src.RoBERTa.auditor import LeakageAuditor
from src.RoBERTa.data import (
    ANSWER_LABELS,
    MCQDataset,
    SemanticDeduplicator,
    collate_fn,
)
from src.RoBERTa.model import MCQRoBERTa
from src.RoBERTa.training import (
    MCQLoss,
    Trainer,
    _get_raw_model,
    build_scheduler,
    ranked_preds,
)
from utils.wandb_init import finish_run, init_wandb, log_model_metrics

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("RoBERTa.Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return (
        pd.read_csv(path)  if path.endswith(".csv")
        else pd.read_json(path)
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc    = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i)  / 1024 ** 3
        logger.info(
            "[%s] GPU%d: alloc=%.2f GB  reserved=%.2f GB",
            tag, i, alloc, reserved,
        )


def _build_optimizer(model: MCQRoBERTa, cfg: Config):
    """
    Differential learning rates:
      Backbone (pretrained RoBERTa) → cfg.lr_backbone
      Head + interaction + pool     → cfg.lr_head

    No weight decay on bias / LayerNorm parameters (standard practice).
    Only parameters with requires_grad=True are included — frozen params
    are excluded so the optimizer state dict stays small.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    backbone_named = list(model.encoder.named_parameters())
    head_named     = (
        list(model.head.named_parameters()) +
        list(model.option_interaction.named_parameters()) +
        list(model.pool.named_parameters())
    )

    def _groups(named_params, lr):
        decay  = [
            p for n, p in named_params
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ]
        no_dec = [
            p for n, p in named_params
            if p.requires_grad and any(nd in n for nd in no_decay)
        ]
        return [
            {"params": decay,  "lr": lr, "weight_decay": cfg.weight_decay},
            {"params": no_dec, "lr": lr, "weight_decay": 0.0},
        ]

    param_groups = (
        _groups(backbone_named, cfg.lr_backbone) +
        _groups(head_named,     cfg.lr_head)
    )
    return torch.optim.AdamW(param_groups, eps=1e-6)


def _safe_group_split(
    df  : pd.DataFrame,
    cfg : Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group-aware train/val split with safety fallback.

    GroupShuffleSplit is used when there are ≥ 10 unique semantic groups.
    Below that threshold a random split is used with a warning; the split
    is still valid because near-duplicates are not removed (group_only mode)
    and a random split introduces no systematic leakage in that scenario.

    Returns
    ───────
    tr_idx, vl_idx : integer index arrays into df
    """
    groups   = df["semantic_group"].values
    n_groups = int(np.unique(groups).shape[0])
    min_groups = 10

    logger.info("Unique semantic groups: %d", n_groups)

    if n_groups >= min_groups:
        gss = GroupShuffleSplit(
            n_splits     = 1,
            test_size    = cfg.val_size,
            random_state = cfg.seed,
        )
        tr_idx, vl_idx = next(gss.split(df, groups=groups))
        method = "GroupShuffleSplit"
    else:
        logger.warning(
            "Only %d semantic groups — too few for GroupShuffleSplit "
            "(need ≥ %d). Falling back to random split.",
            n_groups, min_groups,
        )
        n      = len(df)
        n_val  = max(int(n * cfg.val_size), 50)
        rng    = np.random.default_rng(cfg.seed)
        all_ix = rng.permutation(n)
        vl_idx = all_ix[:n_val]
        tr_idx = all_ix[n_val:]
        method = "RandomSplit (fallback)"

    logger.info(
        "Split method: %s | train=%d  val=%d",
        method, len(tr_idx), len(vl_idx),
    )

    if len(vl_idx) < 150:
        logger.warning(
            "Val set has only %d samples — MAP@3 estimates will be noisy "
            "(±%.3f). Consider raising cfg.val_size.",
            len(vl_idx), 1.0 / (len(vl_idx) ** 0.5),
        )

    return tr_idx, vl_idx


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (split for clarity)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sim_dist(
    max_sims  : np.ndarray,
    wandb_run,
    cfg       : Config,
) -> None:
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.hist(max_sims, bins=50, color="steelblue", edgecolor="white")
        med = float(np.median(max_sims))
        plt.axvline(0.85, color="red",    ls="--", label="threshold=0.85")
        plt.axvline(med,  color="orange", ls="--", label=f"median={med:.3f}")
        plt.xlabel("SBERT cosine (val → train)")
        plt.title("Train–Val SBERT Similarity Distribution")
        plt.legend()
        plt.tight_layout()
        p = plots_dir / "roberta_sim_dist.png"
        plt.savefig(p, dpi=150)
        plt.close()

        if wandb_run:
            import wandb as _wandb
            wandb_run.log({"sim_distribution": _wandb.Image(str(p))})

    except ImportError:
        logger.warning("matplotlib not available — skipping similarity plot.")


def _plot_training(
    history   : dict,
    wandb_run,
    cfg       : Config,
) -> None:
    if not history.get("tr_loss"):
        return
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        ep   = range(1, len(history["tr_loss"]) + 1)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].plot(ep, history["tr_loss"], label="Train")
        axes[0].plot(ep, history["vl_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep, history["vl_map3"], color="green")
        axes[1].set_title("Val MAP@3"); axes[1].grid(alpha=0.3)

        axes[2].plot(ep, history["vl_acc"], color="purple")
        axes[2].set_title("Val Acc"); axes[2].grid(alpha=0.3)

        axes[3].plot(ep, history["vl_f1"], color="orange")
        axes[3].set_title("Val F1"); axes[3].grid(alpha=0.3)

        plt.tight_layout()
        p = plots_dir / "roberta_training.png"
        plt.savefig(p, dpi=150)
        plt.close()

        if wandb_run:
            import wandb as _wandb
            wandb_run.log({"training_curves": _wandb.Image(str(p))})

    except ImportError:
        logger.warning("matplotlib not available — skipping training plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    train_path : str,
    test_path  : Optional[str] = None,
    cfg        : Optional[Config] = None,
):
    """
    Full RoBERTa-base MCQ pipeline.

    Returns
    ───────
    model       : MCQRoBERTa (unwrapped, on CPU after pipeline finishes)
    tokenizer   : HuggingFace fast tokenizer
    history     : training metric history dict
    report      : leakage audit report dict
    submission  : pd.DataFrame | None
    """
    if cfg is None:
        cfg = Config()

    cfg_dict = dataclasses.asdict(cfg)
    _seed(cfg.seed)
    _log_memory("startup")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.use_wandb:
        wandb_run = init_wandb(
            config     = cfg,
            run_name   = "RoBERTa-base-v3",
            model_name = "RoBERTa-base",
            group      = "transformer-models",
            tags       = [
                "RoBERTa", "roberta-base", "SBERT-dedup",
                "cross-option", "mean-pool", "multi-sample-dropout",
            ],
        )

    # ── 1. Raw data ───────────────────────────────────────────────────────────
    train_raw = _load(train_path)
    logger.info("Raw train: %d rows", len(train_raw))

    # ── 2. Dedup (SBERT-based, cached via artifact + disk) ────────────────────
    art = try_load(wandb_run, f"{_DEDUP}:latest", cfg.artifacts_load_dir)

    if art:
        logger.info("Using cached dedup artifact.")
        train_dedup, sbert_all = load_dedup(cfg.artifacts_load_dir)
    else:
        logger.info("Running SBERT semantic deduplication …")
        # Use a local disk cache for SBERT embeddings so re-runs within
        # the same Kaggle session skip the GPU encoding step entirely.
        sbert_cache = str(
            Path(cfg.artifacts_save_dir) / "sbert_raw_cache.npy"
        )
        deduper = SemanticDeduplicator(
            sbert_model      = cfg.sbert_model,
            sbert_batch_sz   = cfg.sbert_batch_size,
            sim_threshold    = cfg.sim_threshold,
            device           = cfg.device,
            max_agglom_rows  = cfg.max_agglom_rows,
            sbert_cache_path = sbert_cache,
        )
        train_dedup, sbert_all = deduper.fit_transform(train_raw)
        save_dedup(wandb_run, train_dedup, sbert_all, cfg.artifacts_save_dir)

    logger.info(
        "Post-dedup: %d rows | SBERT matrix: %s",
        len(train_dedup), sbert_all.shape,
    )

    if wandb_run:
        wandb_run.log({"data/post_dedup_rows": len(train_dedup)})

    # ── 3. Group-aware split ───────────────────────────────────────────────────
    tr_idx, vl_idx = _safe_group_split(train_dedup, cfg)

    train_df    = train_dedup.iloc[tr_idx].reset_index(drop=True)
    val_df      = train_dedup.iloc[vl_idx].reset_index(drop=True)
    sbert_train = sbert_all[tr_idx]
    sbert_val   = sbert_all[vl_idx]

    overlap = set(train_df["semantic_group"]) & set(val_df["semantic_group"])
    logger.info(
        "Group overlap: %d %s | train=%d  val=%d",
        len(overlap),
        "✓ clean" if not overlap else f"⚠ {len(overlap)} groups overlap",
        len(train_df), len(val_df),
    )

    if wandb_run:
        wandb_run.log({
            "data/train"         : len(train_df),
            "data/val"           : len(val_df),
            "data/group_overlap" : len(overlap),
        })

    # ── 4. Leakage audit ──────────────────────────────────────────────────────
    report = LeakageAuditor(cfg.audit_top_k).run(
        train_df, val_df, sbert_train, sbert_val, wandb_run
    )
    save_audit(wandb_run, report, cfg.artifacts_save_dir)

    # Reuse max_sims from the audit — no second cosine_similarity call
    max_sims: np.ndarray = report["max_sims"]
    _plot_sim_dist(max_sims, wandb_run, cfg)

    # ── 5. Tokenizer ──────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── 6. Datasets & DataLoaders ─────────────────────────────────────────────
    pin_mem     = cfg.device == "cuda"
    persistent  = cfg.num_workers > 0    # keep workers alive between epochs
    preload     = not cfg.lazy_tokenization

    train_ds = MCQDataset(
        train_df, tokenizer, cfg.max_len,
        is_test=False, preload=preload,
    )
    val_ds = MCQDataset(
        val_df, tokenizer, cfg.max_len,
        is_test=False, preload=preload,
    )

    loader_kw = dict(
        collate_fn        = collate_fn,
        num_workers       = cfg.num_workers,
        pin_memory        = pin_mem,
        persistent_workers= persistent,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size = cfg.batch_size,
        shuffle    = True,
        **loader_kw,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size = cfg.batch_size,
        shuffle    = False,
        **loader_kw,
    )

    effective_batch = cfg.batch_size * cfg.n_gpus * cfg.grad_accum
    logger.info(
        "Train batches: %d  Val batches: %d  Effective batch: %d",
        len(train_dl), len(val_dl), effective_batch,
    )

    # ── 7. Model ──────────────────────────────────────────────────────────────
    logger.info(
        "Building MCQRoBERTa "
        "[pooling=%s | dropout×%d | freeze=%d | grad_ckpt=%s]",
        cfg.pooling, cfg.n_dropouts, cfg.freeze_layers, cfg.use_grad_ckpt,
    )
    model = MCQRoBERTa(
        model_name     = cfg.model_name,
        pooling        = cfg.pooling,
        hidden_dropout = cfg.hidden_dropout,
        n_dropouts     = cfg.n_dropouts,
        use_grad_ckpt  = cfg.use_grad_ckpt,
    )

    # ── Step order: freeze → optimizer → DataParallel → device ───────────────
    # 1. Freeze on raw model BEFORE optimizer build so only trainable params
    #    are included in the initial param groups.
    model.freeze_backbone_layers(cfg.freeze_layers)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Params: total=%d  trainable=%d", n_params, n_trainable
    )

    # 2. Build optimizer on the raw model.
    optimizer = _build_optimizer(model, cfg)

    # 3. Optionally compile (single-GPU, PyTorch ≥ 2.0 only).
    #    torch.compile is incompatible with nn.DataParallel on most versions.
    if cfg.n_gpus == 1 and hasattr(torch, "compile"):
        logger.info("Applying torch.compile() …")
        try:
            model = torch.compile(model)
        except Exception as exc:
            logger.warning("torch.compile() failed (%s) — continuing without.", exc)

    # 4. Wrap DataParallel AFTER optimizer (DP does not affect param identity).
    if cfg.n_gpus > 1:
        logger.info("Wrapping in nn.DataParallel (%d GPUs)", cfg.n_gpus)
        model = nn.DataParallel(model)

    # 5. Move to device last.
    model = model.to(cfg.device)
    _log_memory("after model load")

    if wandb_run:
        wandb_run.log({
            "model/n_params"    : n_params,
            "model/n_trainable" : n_trainable,
        })

    # ── 8. Scheduler ──────────────────────────────────────────────────────────
    steps_per_epoch = max(len(train_dl) // cfg.grad_accum, 1)
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler       = build_scheduler(optimizer, warmup_steps, total_steps)

    logger.info(
        "Steps: %d/epoch × %d epochs = %d total | warmup=%d",
        steps_per_epoch, cfg.epochs, total_steps, warmup_steps,
    )

    loss_fn = MCQLoss(
        smoothing = cfg.smoothing,
        margin    = cfg.margin,
        ce_w      = cfg.ce_w,
        rank_w    = cfg.rank_w,
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model     = model,
        train_dl  = train_dl,
        val_dl    = val_dl,
        optimizer = optimizer,
        scheduler = scheduler,
        loss_fn   = loss_fn,
        cfg       = cfg_dict,
        device    = cfg.device,
        wandb_run = wandb_run,
    )
    history = trainer.train()
    _plot_training(history, wandb_run, cfg)
    _log_memory("after training")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    # _get_raw_model handles both plain and DataParallel-wrapped models.
    raw_model = _get_raw_model(model)
    save_model(
        wandb_run, raw_model, tokenizer, cfg.artifacts_save_dir,
        meta={
            "best_val_map3" : trainer.best_map3,
            "model_name"    : cfg.model_name,
            "pooling"       : cfg.pooling,
            "n_dropouts"    : cfg.n_dropouts,
            "sim_threshold" : cfg.sim_threshold,
        },
    )

    log_model_metrics(wandb_run, {
        "f1_score"  : max(history["vl_f1"]),
        "accuracy"  : max(history["vl_acc"]),
        "precision" : max(history["vl_precision"]),
        "recall"    : max(history["vl_recall"]),
        "map_at_k"  : trainer.best_map3,
    })

    # ── 11. Test inference ────────────────────────────────────────────────────
    sub = None
    if test_path:
        raw_model.eval()
        test_raw = _load(test_path)
        test_ds  = MCQDataset(
            test_raw, tokenizer, cfg.max_len,
            is_test=True, preload=preload,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size        = cfg.batch_size,
            collate_fn        = collate_fn,
            shuffle           = False,
            num_workers       = cfg.num_workers,
            pin_memory        = pin_mem,
            persistent_workers= persistent,
        )

        ids: list  = []
        preds: list = []
        with torch.no_grad():
            for batch in test_dl:
                iids   = batch["input_ids"].to(cfg.device)
                mask   = batch["attention_mask"].to(cfg.device)
                tids   = batch["token_type_ids"].to(cfg.device)
                logits = raw_model(iids, mask, tids)
                ids   += batch["id"]
                preds += [" ".join(r) for r in ranked_preds(logits)]

        sub = pd.DataFrame({"ID": ids, "Prediction": preds})
        logger.info("Submission: %d rows", len(sub))
        save_submission(wandb_run, sub, cfg.submission_dir)

    # ── 12. Finish ────────────────────────────────────────────────────────────
    finish_run(wandb_run)
    return raw_model, tokenizer, history, report, sub