# pipeline.py
import dataclasses
import logging

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from config.BiLSTM_config import Config
from src.BiLSTM.data import (SemanticDeduplicator, Vocabulary, MCQDataset,
                       collate_fn, normalize_text, ANSWER_LABELS)
from src.BiLSTM.model import MCQBiLSTM
from src.BiLSTM.training  import MCQLoss, Trainer, ranked_preds
from src.BiLSTM.auditor   import LeakageAuditor
from src.BiLSTM.artifacts import (try_load, save_dedup, load_dedup,
                       save_vocab, load_vocab,
                       save_model, save_audit, save_submission,
                       _DEDUP, _VOCAB)
from utils.wandb_init import init_wandb, log_model_metrics, finish_run

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-16s  %(levelname)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str):
    import pandas as pd
    return pd.read_csv(path) if path.endswith('.csv') else pd.read_json(path)


def _plot(history: dict, max_sims: np.ndarray, wandb_run):
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    """Inline plots — not worth a separate file."""
    try:
        import matplotlib.pyplot as plt

        if history:   # training curves
            ep = range(1, len(history['tr_loss']) + 1)
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].plot(ep, history['tr_loss'], label='Train')
            ax[0].plot(ep, history['vl_loss'], label='Val')
            ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[1].plot(ep, history['vl_map3'], color='green')
            ax[1].set_title('Val MAP@3');  ax[1].grid(alpha=.3)
            ax[2].plot(ep, history['vl_acc'],  color='purple')
            ax[2].set_title('Val Acc');    ax[2].grid(alpha=.3)
            plt.tight_layout()
            training_plot = plots_dir / "training_curves.png"
            plt.savefig(training_plot, dpi=150)
            if wandb_run:
                import wandb
                wandb_run.log({
                    "training_curves": wandb.Image(str(training_plot))
                })
            plt.show()

        if max_sims is not None:   # similarity distribution
            plt.figure(figsize=(8, 4))
            plt.hist(max_sims, bins=50, color='steelblue', edgecolor='white')
            plt.axvline(0.85, color='red', ls='--', label='threshold=0.85')
            med = float(np.median(max_sims))
            plt.axvline(med, color='orange', ls='--', label=f'median={med:.3f}')
            plt.xlabel('BoW cosine similarity (val → train)')
            plt.title('Train–Val Similarity Distribution')
            plt.legend(); plt.tight_layout()
            sim_plot = plots_dir / "sim_distribution.png"
            plt.savefig(sim_plot, dpi=150)
            if wandb_run:
                import wandb
                wandb_run.log({
                    "sim_distribution": wandb.Image(str(sim_plot))
                })
            plt.show()

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(train_path: str, test_path: str = None, cfg: Config = None):
    """
    Full pipeline — one call does everything.

    Workflow
    ────────
    Normalize → BoW → Cosine → Cluster → Split
    → Audit → Vocab → DataLoaders → Train → Save → Infer

    Returns
    ───────
    model, vocab, history, report, submission_df
    """
    if cfg is None:
        cfg = Config()

    cfg_dict = dataclasses.asdict(cfg)

    # reproducibility
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(
    config=cfg_dict,
    run_name="BiLSTM",
    model_name="BiLSTM",
    group="scratch-models",
    tags=["BiLSTM", "scratch", "baseline"]
    )

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    train_raw = _load(train_path)
    logger.info(f"Raw train: {len(train_raw):,} rows")

    # ── 2. Deduplicate (BoW-based, cached) ────────────────────────────────────
    #
    #   Workflow:  normalize → BoW vocab → TF-IDF vectors → L2-norm
    #              → AgglomerativeClustering (cosine distance)
    #              → semantic_group column for GroupShuffleSplit
    #
    art = try_load(
        wandb_run,
        f"{_DEDUP}:latest",
        cfg.artifacts_load_dir,
    )

    if art:
        logger.info("Using cached dedup artifact — skipping BoW computation")
        train_dedup, bow_all = load_dedup(cfg.artifacts_load_dir)
    else:
        deduper = SemanticDeduplicator(...)
        train_dedup, bow_all = deduper.fit_transform(train_raw)
        save_dedup(
            wandb_run,
            train_dedup,
            bow_all,
            cfg.artifacts_save_dir,
        )

    # ── 3. Group-aware split ──────────────────────────────────────────────────
    gss = GroupShuffleSplit(1, test_size=cfg.val_size, random_state=cfg.seed)
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
        wandb_run.log({"data/train": len(train_df),
                       "data/val":   len(val_df),
                       "data/group_overlap": len(overlap)})

    # ── 4. Leakage audit (our own cosine) ─────────────────────────────────────
    report = LeakageAuditor(cfg.audit_top_k).run(
        train_df, val_df, bow_train, bow_val, wandb_run)
    save_audit(wandb_run, report, cfg.artifacts_save_dir)

    # sim distribution plot
    max_sims = cosine_similarity(bow_train, bow_val).max(axis=0)
    _plot({}, max_sims, wandb_run)

    # ── 5. LSTM Vocabulary (built from train text, cached) ────────────────────
    art = try_load(
    wandb_run,
    f"{_VOCAB}:latest",
    cfg.artifacts_load_dir
    )
    if art:
        logger.info("Using cached vocab artifact")
        vocab = load_vocab(cfg.artifacts_load_dir)
    else:
        texts = []
        for _, row in train_df.iterrows():
            q = normalize_text(str(row.get('prompt', '')))
            texts.append(q)
            for lbl in ANSWER_LABELS:
                texts.append(
                    f"{q} [SEP] {normalize_text(str(row.get(lbl, '')))}")
        vocab = Vocabulary(cfg.max_vocab, cfg.min_freq).build(texts)
        save_vocab(wandb_run, vocab, cfg.artifacts_save_dir)

    if wandb_run:
        wandb_run.log({"vocab/size": len(vocab)})

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    kw = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    train_dl = DataLoader(MCQDataset(train_df, vocab, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(MCQDataset(val_df,   vocab, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=False, **kw)

    # ── 7. Model (from scratch) ───────────────────────────────────────────────
    model = MCQBiLSTM(
        vocab_size = len(vocab),
        embed_dim  = cfg.embed_dim,
        hidden     = cfg.hidden,
        n_layers   = cfg.n_layers,
        dropout    = cfg.dropout,
        pad_idx    = vocab.PAD_IDX,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")
    if wandb_run:
        wandb_run.log({"model/n_params": n_params})

    # ── 8. Optimizer / scheduler / loss ───────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=cfg.sched_patience, factor=cfg.sched_factor)
    loss_fn = MCQLoss(cfg.smoothing, cfg.margin, cfg.ce_w, cfg.rank_w)

    # ── 9. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_dl, val_dl,
                      opt, sched, loss_fn,
                      cfg.device, cfg_dict, wandb_run)
    history = trainer.train()
    _plot(history, None, wandb_run)

    # ── 10. Save model artifact ───────────────────────────────────────────────
    save_model(wandb_run, model, cfg.artifacts_save_dir,
               meta={"best_val_map3": trainer.best_map3,
                     "vocab_size":    len(vocab)})

    log_model_metrics(
        wandb_run,
        {
            "f1_score": max(history["vl_f1"]),
            "accuracy": max(history["vl_acc"]),
            "precision": max(history["vl_precision"]),
            "recall": max(history["vl_recall"]),
            "map_at_k": trainer.best_map3,
        },
    )
        # ── 11. Test inference ────────────────────────────────────────────────────
    sub = None
    if test_path:
        import pandas as pd
        model.eval()
        test_raw = _load(test_path)
        ds = MCQDataset(test_raw, vocab, cfg.max_len, is_test=True)
        dl = DataLoader(ds, batch_size=cfg.batch_size,
                        collate_fn=collate_fn, shuffle=False)
        ids, preds = [], []
        with torch.no_grad():
            for batch in dl:
                logits = model(batch['options'].to(cfg.device),
                               batch['lengths'].to(cfg.device))
                ids   += batch['id']
                preds += [' '.join(r) for r in ranked_preds(logits)]
        sub = pd.DataFrame({'ID': ids, 'Prediction': preds})
        logger.info(f"Submission: {len(sub):,} rows")
        save_submission(wandb_run, sub, cfg.submission_dir)

    # ── 12. Finish W&B ────────────────────────────────────────────────────────
    finish_run(wandb_run)
    return model, vocab, history, report, sub