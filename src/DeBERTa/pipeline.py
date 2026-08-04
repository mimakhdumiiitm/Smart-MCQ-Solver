# src/DeBERTa/pipeline.py
"""
Full DeBERTa-v3 + SBERT MCQ pipeline.

Workflow
────────
  Load raw data
  → SBERT dedup  (exact MD5 + SBERT cosine + AgglomerativeClustering)
  → Group-aware train/val split
  → SBERT leakage audit
  → DeBERTa tokenization  →  DataLoaders
  → Build MCQDeBERTa  (backbone + cross-option interaction)
  → Differential LR optimizer  (backbone: 8e-6, head: 1e-4)
  → Warmup + cosine scheduler
  → Train  (AMP + grad accum + progressive unfreeze)
  → Save artifacts  →  Test inference  →  Submission
"""

import dataclasses
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.DeBERTa_config import Config
from src.DeBERTa.data import (
    SemanticDeduplicator, MCQDataset, collate_fn,
    normalize_text, ANSWER_LABELS,
)
from src.DeBERTa.embedder  import SBERTEmbedder
from src.DeBERTa.model     import MCQDeBERTa
from src.DeBERTa.training  import MCQLoss, Trainer, build_scheduler, ranked_preds
from src.DeBERTa.auditor   import LeakageAuditor
from src.DeBERTa.artifacts import (
    try_load, save_dedup, load_dedup,
    save_model, save_audit, save_submission,
    _DEDUP,
)
from utils.wandb_init import init_wandb, log_model_metrics, finish_run

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("DeBERTa.Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return (pd.read_csv(path) if path.endswith('.csv')
            else pd.read_json(path))


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: MCQDeBERTa, cfg: Config):
    """
    Differential learning rates.
    Backbone (pretrained DeBERTa) → lr_backbone  (very small, e.g. 8e-6)
    Head + interaction layer      → lr_head       (larger,    e.g. 1e-4)

    Within each group: no weight decay on bias / LayerNorm parameters.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    backbone_named = list(model.encoder.named_parameters())
    head_named     = (
        list(model.head.named_parameters()) +
        list(model.option_interaction.named_parameters()) +
        (list(model.pool.named_parameters())
         if hasattr(model.pool, 'parameters') else [])
    )

    def _groups(named_params, lr):
        decay    = [p for n, p in named_params
                    if p.requires_grad and
                    not any(nd in n for nd in no_decay)]
        no_dec   = [p for n, p in named_params
                    if p.requires_grad and
                    any(nd in n for nd in no_decay)]
        return [
            {"params": decay,  "lr": lr, "weight_decay": cfg.weight_decay},
            {"params": no_dec, "lr": lr, "weight_decay": 0.0},
        ]

    param_groups = (
        _groups(backbone_named, cfg.lr_backbone) +
        _groups(head_named,     cfg.lr_head)
    )
    return torch.optim.AdamW(param_groups, eps=1e-6)


def _plot(history: dict, max_sims: np.ndarray,
          wandb_run, cfg: Config):
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        if history:
            ep  = range(1, len(history['tr_loss']) + 1)
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            axes[0].plot(ep, history['tr_loss'], label='Train')
            axes[0].plot(ep, history['vl_loss'], label='Val')
            axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=.3)
            axes[1].plot(ep, history['vl_map3'], color='green')
            axes[1].set_title('Val MAP@3');  axes[1].grid(alpha=.3)
            axes[2].plot(ep, history['vl_acc'],  color='purple')
            axes[2].set_title('Val Acc');    axes[2].grid(alpha=.3)
            axes[3].plot(ep, history['vl_f1'],   color='orange')
            axes[3].set_title('Val F1');     axes[3].grid(alpha=.3)
            plt.tight_layout()
            p = plots_dir / "deberta_training.png"
            plt.savefig(p, dpi=150); plt.show()
            if wandb_run:
                import wandb
                wandb_run.log({"training_curves": wandb.Image(str(p))})

        if max_sims is not None:
            plt.figure(figsize=(8, 4))
            plt.hist(max_sims, bins=50,
                     color='steelblue', edgecolor='white')
            med = float(np.median(max_sims))
            plt.axvline(0.85, color='red',    ls='--', label='threshold=0.85')
            plt.axvline(med,  color='orange', ls='--',
                        label=f'median={med:.3f}')
            plt.xlabel('SBERT cosine (val → train)')
            plt.title('Train–Val SBERT Similarity Distribution')
            plt.legend(); plt.tight_layout()
            p = plots_dir / "deberta_sim_dist.png"
            plt.savefig(p, dpi=150); plt.show()
            if wandb_run:
                import wandb
                wandb_run.log({"sim_distribution": wandb.Image(str(p))})

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    train_path : str,
    test_path  : str    = None,
    cfg        : Config = None,
):
    """
    Full DeBERTa-v3 MCQ pipeline.

    Returns
    ───────
    model, tokenizer, history, report, submission_df
    """
    if cfg is None:
        cfg = Config()

    cfg_dict = dataclasses.asdict(cfg)
    _seed(cfg.seed)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(
        config     = cfg,
        run_name   = "DeBERTa-v3-small-SBERT",
        model_name = "DeBERTa-v3",
        group      = "transformer-models",
        tags       = ["DeBERTa", "deberta-v3-small",
                      "SBERT-dedup", "cross-option"],
    )

    # ── 1. Raw data ───────────────────────────────────────────────────────────
    train_raw = _load(train_path)
    logger.info(f"Raw train: {len(train_raw):,} rows")

    # ── 2. Dedup (SBERT-based, cached) ────────────────────────────────────────
    art = try_load(wandb_run, f"{_DEDUP}:latest", cfg.artifacts_load_dir)

    if art:
        logger.info("Using cached dedup artifact")
        train_dedup, sbert_all = load_dedup(cfg.artifacts_load_dir)
    else:
        logger.info("Running SBERT semantic deduplication …")
        deduper = SemanticDeduplicator(
            sbert_model    = cfg.sbert_model,
            sbert_batch_sz = cfg.sbert_batch_size,
            sim_threshold  = cfg.sim_threshold,
            device         = cfg.device,
        )
        train_dedup, sbert_all = deduper.fit_transform(train_raw)
        save_dedup(wandb_run, train_dedup, sbert_all, cfg.artifacts_save_dir)

    logger.info(f"Post-dedup: {len(train_dedup):,} rows | "
                f"SBERT matrix: {sbert_all.shape}")

    # ── 3. Group-aware split ──────────────────────────────────────────────────
    gss = GroupShuffleSplit(1, test_size=cfg.val_size,
                            random_state=cfg.seed)
    tr_idx, vl_idx = next(
        gss.split(train_dedup, groups=train_dedup['semantic_group']))

    train_df    = train_dedup.iloc[tr_idx].reset_index(drop=True)
    val_df      = train_dedup.iloc[vl_idx].reset_index(drop=True)
    sbert_train = sbert_all[tr_idx]
    sbert_val   = sbert_all[vl_idx]

    overlap = (set(train_df['semantic_group']) &
               set(val_df['semantic_group']))
    logger.info(
        f"Group overlap: {len(overlap)} "
        f"{'✓ clean' if not overlap else '✗ LEAKAGE'} | "
        f"train={len(train_df):,}  val={len(val_df):,}"
    )

    if wandb_run:
        wandb_run.log({
            "data/train"         : len(train_df),
            "data/val"           : len(val_df),
            "data/group_overlap" : len(overlap),
        })

    # AFTER the split, ADD
    if len(val_df) < 200:
        logger.warning(
            f"Validation set has only {len(val_df)} samples. "
            f"MAP@3 estimates will be noisy. "
            f"Consider raising cfg.val_size or loosening sim_threshold."
        )
    # ── 4. Leakage audit ──────────────────────────────────────────────────────
    report = LeakageAuditor(cfg.audit_top_k).run(
        train_df, val_df, sbert_train, sbert_val, wandb_run)
    save_audit(wandb_run, report, cfg.artifacts_save_dir)

    max_sims = cosine_similarity(sbert_train, sbert_val).max(axis=0)
    _plot({}, max_sims, wandb_run, cfg)

    # ── 5. Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading DeBERTa tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── 6. Datasets & DataLoaders ─────────────────────────────────────────────
    kw = dict(
        collate_fn  = collate_fn,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device == "cuda"),
    )
    train_ds = MCQDataset(train_df, tokenizer, cfg.max_len, is_test=False)
    val_ds   = MCQDataset(val_df,   tokenizer, cfg.max_len, is_test=False)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw)

    logger.info(f"Train batches: {len(train_dl)}  "
                f"Val batches: {len(val_dl)}")

    # ── 7. Model ──────────────────────────────────────────────────────────────
    logger.info(
        f"Building MCQDeBERTa  "
        f"[pooling={cfg.pooling} | grad_ckpt={cfg.use_grad_ckpt}]"
    )
    model = MCQDeBERTa(
        model_name     = cfg.model_name,
        pooling        = cfg.pooling,
        hidden_dropout = cfg.hidden_dropout,
        use_grad_ckpt  = cfg.use_grad_ckpt,
    )
    print("Model dtype check:")
    print("rel_embeddings:",model.encoder.encoder.rel_embeddings.weight.dtype)

    print("embeddings:",
        next(model.encoder.embeddings.parameters()).dtype)

    print("first encoder layer:",
        next(model.encoder.encoder.layer[0].parameters()).dtype)
    model.freeze_backbone_layers(cfg.freeze_layers)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    logger.info(f"Params total={n_params:,}  trainable={n_trainable:,}")

    if wandb_run:
        wandb_run.log({
            "model/n_params"    : n_params,
            "model/n_trainable" : n_trainable,
        })

    # ── 8. Optimizer + scheduler ──────────────────────────────────────────────
    optimizer = _build_optimizer(model, cfg)

    steps_per_epoch = max(len(train_dl) // cfg.grad_accum, 1)
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler       = build_scheduler(optimizer, warmup_steps, total_steps)

    logger.info(
        f"Scheduler: {warmup_steps} warmup / {total_steps} total steps"
    )

    loss_fn = MCQLoss(
        smoothing = cfg.smoothing,
        margin    = cfg.margin,
        ce_w      = cfg.ce_w,
        rank_w    = cfg.rank_w,
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model      = model,
        train_dl   = train_dl,
        val_dl     = val_dl,
        optimizer  = optimizer,
        scheduler  = scheduler,
        loss_fn    = loss_fn,
        cfg        = cfg_dict,
        device     = cfg.device,
        wandb_run  = wandb_run,
    )
    history = trainer.train()
    _plot(history, None, wandb_run, cfg)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    save_model(
        wandb_run, model, tokenizer, cfg.artifacts_save_dir,
        meta={
            "best_val_map3" : trainer.best_map3,
            "model_name"    : cfg.model_name,
            "pooling"       : cfg.pooling,
            "sbert_model"   : cfg.sbert_model,
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
        model.eval()
        test_raw = _load(test_path)
        test_ds  = MCQDataset(
            test_raw, tokenizer, cfg.max_len, is_test=True)
        test_dl  = DataLoader(
            test_ds, batch_size=cfg.batch_size,
            collate_fn=collate_fn, shuffle=False,
            num_workers=cfg.num_workers)

        ids, preds = [], []
        with torch.no_grad():
            for batch in test_dl:
                iids   = batch['input_ids'].to(cfg.device)
                mask   = batch['attention_mask'].to(cfg.device)
                tids   = batch['token_type_ids'].to(cfg.device)
                logits = model(iids, mask, tids)
                ids   += batch['id']
                preds += [' '.join(r) for r in ranked_preds(logits)]

        sub = pd.DataFrame({'ID': ids, 'Prediction': preds})
        logger.info(f"Submission: {len(sub):,} rows")
        save_submission(wandb_run, sub, cfg.submission_dir)

    # ── 12. Finish ────────────────────────────────────────────────────────────
    finish_run(wandb_run)
    return model, tokenizer, history, report, sub