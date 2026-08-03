# pipeline.py
import dataclasses
import logging

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from config.BiLSTM_config  import Config
from src.BiLSTM.data import (SemanticDeduplicator, Vocabulary, MCQDataset,
                      collate_fn, normalize_text, ANSWER_LABELS)
from src.BiLSTM.model    import MCQBiLSTM
from src.BiLSTM.training import MCQLoss, Trainer, ranked_preds
from src.BiLSTM.auditor  import LeakageAuditor
from src.BiLSTM.artifacts import (try_load, save_dedup, load_dedup,
                       save_vocab, load_vocab,
                       save_model, save_audit, save_submission,
                       _DEDUP, _VOCAB)

from utils.wandb_init import init_wandb, log_model_metrics, finish_run
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-16s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


def _load(path: str):
    import pandas as pd
    return pd.read_csv(path) if path.endswith('.csv') else pd.read_json(path)


def _plot(history, max_sims, wandb_run):

    # Similarity histogram
    if max_sims is not None:
        plt.figure(figsize=(6,4))
        plt.hist(max_sims, bins=50)
        plt.title("Train-Val Similarity")
        plt.show()

    # Training curves
    if history and "tr_loss" in history:
        ep = range(1, len(history["tr_loss"]) + 1)

        fig, ax = plt.subplots(1,3, figsize=(15,4))

        ax[0].plot(ep, history["tr_loss"], label="Train")
        ax[0].plot(ep, history["va_loss"], label="Val")
        ax[0].legend()

        ax[1].plot(ep, history["va_map"])

        plt.show()


# ─────────────────────────────────────────────────────────────────────────────

def run(train_path: str, test_path: str = None, cfg: Config = None):
    if cfg is None:
        cfg = Config()

    cfg_dict = dataclasses.asdict(cfg)

    # seed
    import random, numpy as np, torch
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(cfg, run_name=cfg.wandb_run_name, model_tag="bilstm")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    train_raw = _load(train_path)
    logger.info(f"Raw train: {len(train_raw):,} rows")

    # ── 2. Dedup (cached) ─────────────────────────────────────────────────────
    art = try_load(wandb_run, f"{_DEDUP}:latest", cfg.artifact_dir)
    if art:
        train_dedup, embs_all = load_dedup(cfg.artifact_dir)
    else:
        deduper = SemanticDeduplicator(cfg.sbert_model, cfg.sim_threshold)
        train_dedup, embs_all = deduper.fit_transform(train_raw)
        save_dedup(wandb_run, train_dedup, embs_all, cfg.artifact_dir)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    gss = GroupShuffleSplit(1, test_size=cfg.val_size, random_state=cfg.seed)
    tr_idx, vl_idx = next(
        gss.split(train_dedup, groups=train_dedup['semantic_group']))

    train_df   = train_dedup.iloc[tr_idx].reset_index(drop=True)
    val_df     = train_dedup.iloc[vl_idx].reset_index(drop=True)
    embs_train = embs_all[tr_idx]
    embs_val   = embs_all[vl_idx]

    overlap = set(train_df['semantic_group']) & set(val_df['semantic_group'])
    logger.info(f"Group overlap: {len(overlap)} "
                f"{'✓ clean' if not overlap else '✗ LEAKAGE'} | "
                f"train={len(train_df):,} val={len(val_df):,}")

    if wandb_run:
        wandb_run.log({"data/train": len(train_df), "data/val": len(val_df),
                       "data/group_overlap": len(overlap)})

    # ── 4. Audit ──────────────────────────────────────────────────────────────
    report = LeakageAuditor(cfg.audit_top_k).run(
        train_df, val_df, embs_train, embs_val, wandb_run)
    save_audit(wandb_run, report, cfg.artifact_dir)

    max_sims = (embs_train @ embs_val.T).max(axis=0)
    _plot({}, max_sims, wandb_run)   # sim dist only at this stage

    # ── 5. Vocabulary (cached) ────────────────────────────────────────────────
    art = try_load(wandb_run, f"{_VOCAB}:latest", cfg.artifact_dir)
    if art:
        vocab = load_vocab(cfg.artifact_dir)
    else:
        texts = []
        for _, row in train_df.iterrows():
            q = normalize_text(str(row.get('prompt', '')))
            texts.append(q)
            for lbl in ANSWER_LABELS:
                texts.append(f"{q} [SEP] {normalize_text(str(row.get(lbl, '')))}")
        vocab = Vocabulary(cfg.max_vocab, cfg.min_freq).build(texts)
        save_vocab(wandb_run, vocab, cfg.artifact_dir)

    if wandb_run:
        wandb_run.log({"vocab/size": len(vocab)})

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    kw = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    train_dl = DataLoader(MCQDataset(train_df, vocab, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(MCQDataset(val_df,   vocab, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=False, **kw)

    # ── 7. Model ──────────────────────────────────────────────────────────────
    model = MCQBiLSTM(len(vocab), cfg.embed_dim, cfg.hidden,
                      cfg.n_layers, cfg.dropout, vocab.PAD_IDX)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")
    if wandb_run:
        wandb_run.log({"model/n_params": n_params})

    # ── 8. Train ──────────────────────────────────────────────────────────────
    opt     = torch.optim.AdamW(model.parameters(),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=cfg.sched_patience, factor=cfg.sched_factor)
    loss_fn = MCQLoss(cfg.smoothing, cfg.margin, cfg.ce_w, cfg.rank_w)

    trainer = Trainer(model, train_dl, val_dl, opt, sched,
                      loss_fn, cfg.device, cfg_dict, wandb_run)
    history = trainer.train()

    _plot(history, max_sims, wandb_run)   # full training curves

    # ── 9. Save model ─────────────────────────────────────────────────────────
    save_model(wandb_run, model, cfg.artifact_dir,
               meta={"best_val_map3": trainer.best_map3,
                     "vocab_size": len(vocab)})

    log_model_metrics(wandb_run, {
        "f1_score" : trainer.best_map3,
        "accuracy" : max(history['vl_acc']),
        "precision": trainer.best_map3,
        "recall"   : trainer.best_map3,
        "map_at_k" : trainer.best_map3,
    })

    # ── 10. Inference ─────────────────────────────────────────────────────────
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
        logger.info(f"Submission: {len(sub)} rows")
        save_submission(wandb_run, sub, cfg.artifact_dir)

    # ── 11. Finish ────────────────────────────────────────────────────────────
    finish_run(wandb_run)
    return model, vocab, history, report, sub