# src/DeBERTa/pipeline.py
"""
DeBERTa-v3 full pipeline — mirrors BiLSTM pipeline.py exactly.

Workflow
────────
Normalize → BoW → Cosine → Cluster → Split
→ Audit → Tokenizer → DataLoaders → Fine-tune → Save → Infer
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
from transformers import get_linear_schedule_with_warmup

from config.DeBERTa_config import Config
from src.BiLSTM.data import (
    SemanticDeduplicator,
    normalize_text,
    ANSWER_LABELS,
)
from src.BiLSTM.auditor  import LeakageAuditor
from src.BiLSTM.training import MCQLoss
from src.BiLSTM.artifacts import _DEDUP

from src.DeBERTa.data      import MCQDataset, collate_fn, load_tokenizer
from src.DeBERTa.model     import MCQDeBERTa
from src.DeBERTa.training  import Trainer
from src.DeBERTa.artifacts import (
    try_load, save_dedup, load_dedup,
    save_tokenizer, save_model,
    save_audit, save_submission,
    _MODEL,
)
from utils.wandb_init import init_wandb, log_model_metrics, finish_run

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-18s  %(levelname)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("DeBERTa.Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return (pd.read_csv(path)
            if path.endswith('.csv')
            else pd.read_json(path))


def _plot(history: dict, max_sims: np.ndarray,
          wandb_run, cfg: Config):
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        # ── training curves ──────────────────────────────────────────────────
        if history:
            ep  = range(1, len(history['tr_loss']) + 1)
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].plot(ep, history['tr_loss'], label='Train')
            ax[0].plot(ep, history['vl_loss'], label='Val')
            ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[1].plot(ep, history['vl_map3'], color='green')
            ax[1].set_title('Val MAP@3'); ax[1].grid(alpha=.3)
            ax[2].plot(ep, history['vl_acc'],  color='purple')
            ax[2].set_title('Val Acc');   ax[2].grid(alpha=.3)
            plt.tight_layout()
            out = plots_dir / "deberta_training_curves.png"
            plt.savefig(out, dpi=150)
            if wandb_run:
                import wandb
                wandb_run.log({"training_curves": wandb.Image(str(out))})
            plt.show()

        # ── similarity distribution ──────────────────────────────────────────
        if max_sims is not None:
            plt.figure(figsize=(8, 4))
            plt.hist(max_sims, bins=50,
                     color='steelblue', edgecolor='white')
            plt.axvline(0.85, color='red', ls='--', label='threshold=0.85')
            med = float(np.median(max_sims))
            plt.axvline(med, color='orange', ls='--',
                        label=f'median={med:.3f}')
            plt.xlabel('BoW cosine similarity (val → train)')
            plt.title('Train–Val Similarity Distribution')
            plt.legend(); plt.tight_layout()
            out = plots_dir / "deberta_sim_distribution.png"
            plt.savefig(out, dpi=150)
            if wandb_run:
                import wandb
                wandb_run.log({"sim_distribution": wandb.Image(str(out))})
            plt.show()

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(train_path: str,
        test_path : str    = None,
        cfg       : Config = None):
    """
    Full DeBERTa-v3 fine-tuning pipeline.

    Returns
    ───────
    model, tokenizer, history, report, submission_df
    """
    if cfg is None:
        cfg = Config()

    cfg_dict = dataclasses.asdict(cfg)

    # ── reproducibility ───────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(
        config     = cfg,
        run_name   = "DeBERTa-v3",
        model_name = "DeBERTa-v3",
        group      = "pretrained-models",
        tags       = ["DeBERTa", "deberta-v3", "fine-tune"],
    )

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    train_raw = _load(train_path)
    logger.info(f"Raw train: {len(train_raw):,} rows")

    # ── 2. Deduplicate (BoW-based, cached) ────────────────────────────────────
    art = try_load(wandb_run, f"{_DEDUP}:latest", cfg.artifacts_load_dir)

    if art:
        logger.info("Using cached dedup artifact")
        train_dedup, bow_all = load_dedup(cfg.artifacts_load_dir)
    else:
        deduper = SemanticDeduplicator(
            sim_threshold = cfg.sim_threshold,
            max_features  = cfg.bow_max_features,
            ngram_max     = cfg.bow_ngram_max,
        )
        train_dedup, bow_all = deduper.fit_transform(train_raw)
        save_dedup(wandb_run, train_dedup, bow_all, cfg.artifacts_save_dir)

    # ── 3. Group-aware split ──────────────────────────────────────────────────
    gss = GroupShuffleSplit(1, test_size=cfg.val_size,
                            random_state=cfg.seed)
    tr_idx, vl_idx = next(
        gss.split(train_dedup, groups=train_dedup['semantic_group']))

    train_df  = train_dedup.iloc[tr_idx].reset_index(drop=True)
    val_df    = train_dedup.iloc[vl_idx].reset_index(drop=True)
    bow_train = bow_all[tr_idx]
    bow_val   = bow_all[vl_idx]

    overlap = (set(train_df['semantic_group']) &
               set(val_df['semantic_group']))
    logger.info(f"Group overlap: {len(overlap)} "
                f"{'✓ clean' if not overlap else '✗ LEAKAGE'} | "
                f"train={len(train_df):,}  val={len(val_df):,}")

    if wandb_run:
        wandb_run.log({
            "data/train"         : len(train_df),
            "data/val"           : len(val_df),
            "data/group_overlap" : len(overlap),
        })

    # ── 4. Leakage audit ──────────────────────────────────────────────────────
    report   = LeakageAuditor(cfg.audit_top_k).run(
        train_df, val_df, bow_train, bow_val, wandb_run)
    save_audit(wandb_run, report, cfg.artifacts_save_dir)

    max_sims = cosine_similarity(bow_train, bow_val).max(axis=0)
    _plot({}, max_sims, wandb_run, cfg)

    # ── 5. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg.model_name)
    save_tokenizer(wandb_run, tokenizer,
                   cfg.artifacts_save_dir, cfg.model_name)

    if wandb_run:
        wandb_run.log({"tokenizer/vocab_size": tokenizer.vocab_size})

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    kw = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)

    train_dl = DataLoader(
        MCQDataset(train_df, tokenizer, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=True, **kw)

    val_dl = DataLoader(
        MCQDataset(val_df, tokenizer, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=False, **kw)

    # ── 7. Model ──────────────────────────────────────────────────────────────
    model = MCQDeBERTa(
        model_name         = cfg.model_name,
        classifier_dropout = cfg.classifier_dropout,
        freeze_layers      = cfg.freeze_layers,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {n_params:,}")
    if wandb_run:
        wandb_run.log({"model/n_params": n_params})

    # ── 8. Optimizer & scheduler ──────────────────────────────────────────────
    # Separate LR for encoder vs. head — standard fine-tuning practice
    encoder_params = [
        p for n, p in model.named_parameters()
        if 'encoder' in n and p.requires_grad]
    head_params = [
        p for n, p in model.named_parameters()
        if 'head' in n and p.requires_grad]

    opt = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.lr},
        {"params": head_params,    "lr": cfg.lr * 10},   # head trains faster
    ], weight_decay=cfg.weight_decay)

    # total optimizer steps (accounting for grad accumulation)
    total_steps  = (len(train_dl) // cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    logger.info(f"Scheduler: linear warmup {warmup_steps} / {total_steps} steps")

    loss_fn = MCQLoss(
        smoothing = cfg.smoothing,
        margin    = cfg.margin,
        ce_w      = cfg.ce_w,
        rank_w    = cfg.rank_w,
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, train_dl, val_dl,
        opt, sched, loss_fn,
        cfg.device, cfg_dict, wandb_run,
    )
    history = trainer.train()
    _plot(history, None, wandb_run, cfg)

    # ── 10. Save model ────────────────────────────────────────────────────────
    save_model(wandb_run, model, cfg.artifacts_save_dir,
               meta={"best_val_map3": trainer.best_map3,
                     "model_name":    cfg.model_name})

    log_model_metrics(
        wandb_run,
        {
            "f1_score" : max(history["vl_f1"]),
            "accuracy" : max(history["vl_acc"]),
            "precision": max(history["vl_precision"]),
            "recall"   : max(history["vl_recall"]),
            "map_at_k" : trainer.best_map3,
        },
    )

    # ── 11. Test inference ────────────────────────────────────────────────────
    sub = None
    if test_path:
        from src.BiLSTM.training import ranked_preds

        model.eval()
        test_raw = _load(test_path)
        ds = MCQDataset(test_raw, tokenizer, cfg.max_len, is_test=True)
        dl = DataLoader(ds, batch_size=cfg.batch_size,
                        collate_fn=collate_fn, shuffle=False)

        ids, preds = [], []
        with torch.no_grad():
            for batch in dl:
                logits = model(
                    batch['input_ids']     .to(cfg.device),
                    batch['attention_mask'].to(cfg.device),
                )
                ids   += batch['id']
                preds += [' '.join(r) for r in ranked_preds(logits)]

        sub = pd.DataFrame({'ID': ids, 'Prediction': preds})
        logger.info(f"Submission: {len(sub):,} rows")
        save_submission(wandb_run, sub, cfg.submission_dir)

    # ── 12. Finish ────────────────────────────────────────────────────────────
    finish_run(wandb_run)
    return model, tokenizer, history, report, sub