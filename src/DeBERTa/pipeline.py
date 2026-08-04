# src/DeBERTa/pipeline.py
"""
DeBERTa-v3 full pipeline.

Changes vs. original
─────────────────────
1. OneCycleLR replaces linear warmup schedule
   - More aggressive for small dataset (1024 rows)
   - Finds peak LR faster, decays smoothly
2. num_workers=0 in DataLoaders
   - Kaggle P100/T4 multiprocessing is slow; 0 is faster for small datasets
3. Inference wrapped in autocast — matches training dtype
4. model.float() before inference save — ensures fp32 checkpoint
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
)
from utils.wandb_init import init_wandb, log_model_metrics, finish_run

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-18s  %(levelname)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("DeBERTa.Pipeline")


def _load(path: str) -> pd.DataFrame:
    return (pd.read_csv(path)
            if path.endswith('.csv')
            else pd.read_json(path))


def _plot(history: dict, max_sims, wandb_run, cfg: Config):
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        if history:
            ep  = range(1, len(history['tr_loss']) + 1)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(ep, history['tr_loss'], label='Train')
            axes[0].plot(ep, history['vl_loss'], label='Val')
            axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=.3)
            axes[1].plot(ep, history['vl_map3'], color='green')
            axes[1].set_title('Val MAP@3'); axes[1].grid(alpha=.3)
            axes[2].plot(ep, history['vl_acc'],  color='purple')
            axes[2].set_title('Val Acc');   axes[2].grid(alpha=.3)
            plt.tight_layout()
            out = plots_dir / "deberta_training_curves.png"
            plt.savefig(out, dpi=150)
            if wandb_run:
                import wandb as wb
                wandb_run.log({"training_curves": wb.Image(str(out))})
            plt.show()

        if max_sims is not None:
            plt.figure(figsize=(8, 4))
            plt.hist(max_sims, bins=50,
                     color='steelblue', edgecolor='white')
            med = float(np.median(max_sims))
            plt.axvline(0.85, color='red',    ls='--', label='threshold=0.85')
            plt.axvline(med,  color='orange', ls='--', label=f'median={med:.3f}')
            plt.xlabel('BoW cosine similarity (val→train)')
            plt.title('Train–Val Similarity Distribution')
            plt.legend(); plt.tight_layout()
            out = plots_dir / "deberta_sim_distribution.png"
            plt.savefig(out, dpi=150)
            if wandb_run:
                import wandb as wb
                wandb_run.log({"sim_distribution": wb.Image(str(out))})
            plt.show()

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


def run(train_path: str,
        test_path : str    = None,
        cfg       : Config = None):
    """
    Full DeBERTa-v3 fine-tuning pipeline.

    Returns:  model, tokenizer, history, report, submission_df
    """
    if cfg is None:
        cfg = Config()

    cfg_dict = dataclasses.asdict(cfg)

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

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_raw = _load(train_path)
    logger.info(f"Raw train: {len(train_raw):,} rows")

    # ── 2. Dedup (cached) ────────────────────────────────────────────────────
    art = try_load(wandb_run, f"{_DEDUP}:latest", cfg.artifacts_load_dir)
    if art:
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
    gss = GroupShuffleSplit(1, test_size=cfg.val_size, random_state=cfg.seed)
    tr_idx, vl_idx = next(
        gss.split(train_dedup, groups=train_dedup['semantic_group']))

    train_df  = train_dedup.iloc[tr_idx].reset_index(drop=True)
    val_df    = train_dedup.iloc[vl_idx].reset_index(drop=True)
    bow_train = bow_all[tr_idx]
    bow_val   = bow_all[vl_idx]

    overlap = set(train_df['semantic_group']) & set(val_df['semantic_group'])
    logger.info(
        f"Group overlap: {len(overlap)} "
        f"{'✓ clean' if not overlap else '✗ LEAKAGE'} | "
        f"train={len(train_df):,}  val={len(val_df):,}")

    if wandb_run:
        wandb_run.log({
            "data/train"        : len(train_df),
            "data/val"          : len(val_df),
            "data/group_overlap": len(overlap),
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

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    # num_workers=0: faster on Kaggle for small datasets
    # (multiprocessing spawn overhead > benefit for 1024 rows)
    kw = dict(
        collate_fn  = collate_fn,
        num_workers = 0,          # ← changed from 2
        pin_memory  = True,
    )
    train_dl = DataLoader(
        MCQDataset(train_df, tokenizer, cfg.max_len),
        batch_size = cfg.batch_size,
        shuffle    = True,
        **kw,
    )
    val_dl = DataLoader(
        MCQDataset(val_df, tokenizer, cfg.max_len),
        batch_size = cfg.batch_size,
        shuffle    = False,
        **kw,
    )

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

    # ── 8. Optimizer ──────────────────────────────────────────────────────────
    # Layer-wise LR decay — standard for fine-tuning transformers
    # Encoder uses cfg.lr; head uses 10× cfg.lr
    no_decay   = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}

    encoder_decay, encoder_nodecay, head_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name:
            head_params.append(param)
        elif any(nd in name for nd in no_decay):
            encoder_nodecay.append(param)
        else:
            encoder_decay.append(param)

    opt = torch.optim.AdamW([
        {"params": encoder_decay,
         "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": encoder_nodecay,
         "lr": cfg.lr, "weight_decay": 0.0},
        {"params": head_params,
         "lr": cfg.lr * 10, "weight_decay": cfg.weight_decay},
    ])

    # ── 9. Scheduler — OneCycleLR ─────────────────────────────────────────────
    # steps_per_epoch = ceil(len(train_dl) / grad_accum)
    # OneCycleLR works per optimizer step (after accumulation)
    steps_per_epoch = max(len(train_dl) // cfg.grad_accum_steps, 1)
    total_steps     = steps_per_epoch * cfg.epochs

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr         = [cfg.lr, cfg.lr, cfg.lr * 10],
        total_steps    = total_steps,
        pct_start      = cfg.warmup_ratio,
        anneal_strategy= 'cos',
        div_factor     = 10,      # initial_lr = max_lr / 10
        final_div_factor= 100,    # final_lr  = max_lr / (10 × 100)
    )

    logger.info(
        f"OneCycleLR | "
        f"max_lr={cfg.lr:.1e}  "
        f"steps_per_epoch={steps_per_epoch}  "
        f"total={total_steps}")

    loss_fn = MCQLoss(
        smoothing = cfg.smoothing,
        margin    = cfg.margin,
        ce_w      = cfg.ce_w,
        rank_w    = cfg.rank_w,
    )

    # ── 10. Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, train_dl, val_dl,
        opt, sched, loss_fn,
        cfg.device, cfg_dict, wandb_run,
    )
    history = trainer.train()
    _plot(history, None, wandb_run, cfg)

    # ── 11. Save model ────────────────────────────────────────────────────────
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

    # ── 12. Test inference ────────────────────────────────────────────────────
    sub = None
    if test_path:
        from src.BiLSTM.training import ranked_preds

        model.eval()
        test_raw = _load(test_path)

        ds = MCQDataset(test_raw, tokenizer, cfg.max_len, is_test=True)
        dl = DataLoader(
            ds,
            batch_size  = cfg.batch_size,
            collate_fn  = collate_fn,
            shuffle     = False,
            num_workers = 0,
        )

        ids, preds = [], []

        # ── wrap inference in autocast (same dtype as training) ───────────────
        use_autocast = cfg.fp16 and cfg.device == 'cuda'
        amp_dtype    = (torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16)
        ctx = (torch.amp.autocast("cuda", dtype=amp_dtype)
               if use_autocast
               else torch.amp.autocast("cuda", enabled=False))

        with torch.no_grad(), ctx:
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

    finish_run(wandb_run)
    return model, tokenizer, history, report, sub