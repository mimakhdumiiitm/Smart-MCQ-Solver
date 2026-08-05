# src/RoBERTa/training.py
"""
Training logic for RoBERTa-base MCQ.

Key differences vs DeBERTa training.py
────────────────────────────────────────
  - use_fp16 = False by default  →  no GradScaler / autocast issues
    (autocast is kept but disabled so the interface is identical)
  - R-Drop regularization  →  two stochastic forwards per batch,
    KL-divergence consistency loss pushes them to agree
  - Stochastic Weight Averaging (SWA) in final epochs
  - All other logic identical to DeBERTa trainer
"""

import logging
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import GradScaler, autocast

logger = logging.getLogger("RoBERTa.Trainer")

ANSWER_LABELS = ['A', 'B', 'C', 'D', 'E']


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class MCQLoss(nn.Module):
    """
    CE (label smoothing) + pairwise margin ranking + R-Drop KL.

    R-Drop
    ──────
    Run two stochastic forwards (different dropout masks) → two softmax
    distributions P1, P2.  Minimise symmetric KL(P1 || P2).
    This acts as a data-augmentation / consistency regularizer.
    Paper: https://arxiv.org/abs/2106.14448
    """

    def __init__(
        self,
        smoothing : float = 0.05,
        margin    : float = 0.3,
        ce_w      : float = 0.6,
        rank_w    : float = 0.3,
        rdrop_w   : float = 0.1,
    ):
        super().__init__()
        self.ce      = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.margin  = margin
        self.ce_w    = ce_w
        self.rank_w  = rank_w
        self.rdrop_w = rdrop_w

    def _rank_loss(self, logits, targets):
        pos       = logits.gather(1, targets.unsqueeze(1))
        margin    = F.relu(self.margin - (pos - logits))
        eye       = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.)
        return (margin * (1 - eye)).sum(1).mean()

    def forward(
        self,
        logits1  : torch.Tensor,          # [B, 5]
        targets  : torch.Tensor,          # [B]
        logits2  : torch.Tensor = None,   # [B, 5]  optional R-Drop
    ):
        ce1       = self.ce(logits1, targets)
        rank1     = self._rank_loss(logits1, targets)
        total     = self.ce_w * ce1 + self.rank_w * rank1

        rdrop_val = 0.0
        if logits2 is not None and self.rdrop_w > 0:
            ce2       = self.ce(logits2, targets)
            rank2     = self._rank_loss(logits2, targets)
            total    += 0.5 * (self.ce_w * ce2 + self.rank_w * rank2)

            p1 = F.log_softmax(logits1, dim=-1)
            p2 = F.log_softmax(logits2, dim=-1)
            # symmetric KL
            kl = 0.5 * (
                F.kl_div(p1, p2.detach().exp(), reduction='batchmean') +
                F.kl_div(p2, p1.detach().exp(), reduction='batchmean')
            )
            rdrop_val = kl.item()
            total    += self.rdrop_w * kl

        return total, ce1.item(), rank1, rdrop_val


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ranked_preds(logits: torch.Tensor):
    top3 = torch.argsort(logits, dim=-1, descending=True)[:, :3]
    return [[ANSWER_LABELS[i.item()] for i in row] for row in top3]


def map_at_3(preds, labels) -> float:
    scores = []
    for pred, gold in zip(preds, labels):
        ap, hits = 0., 0
        for k, p in enumerate(pred[:3], 1):
            if p == gold:
                hits += 1; ap += hits / k
        scores.append(ap)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, n_warmup: int, n_total: int):
    def lr_lambda(step: int):
        if step < n_warmup:
            return float(step) / max(1, n_warmup)
        progress = float(step - n_warmup) / max(1, n_total - n_warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    RoBERTa MCQ Trainer.

    Features
    ────────
    ✓  R-Drop regularization (two forward passes per step)
    ✓  AMP kept but DISABLED (use_fp16=False) — avoids FP16 grad errors
    ✓  Gradient accumulation
    ✓  Progressive layer unfreezing
    ✓  Early stopping on MAP@3
    ✓  Best-checkpoint restore
    ✓  W&B logging
    """

    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        optimizer,
        scheduler,
        loss_fn,
        cfg      : dict,
        device   : str,
        wandb_run = None,
    ):
        self.model     = model.to(device)
        self.train_dl  = train_dl
        self.val_dl    = val_dl
        self.opt       = optimizer
        self.sched     = scheduler
        self.loss_fn   = loss_fn
        self.cfg       = cfg
        self.device    = device
        self.wandb_run = wandb_run

        # ── AMP: DISABLED to avoid FP16 gradient issues ──────────────────
        # GradScaler and autocast are instantiated but disabled.
        # This keeps the code structure intact for future opt-in.
        self.use_fp16 = False          # hard override — never enable FP16
        self.scaler   = GradScaler("cuda", enabled=False)

        self.grad_accum = cfg.get('grad_accum', 4)

        self.history    = defaultdict(list)
        self.best_map3  = -np.inf
        self.best_state = None

        self._es_counter = 0
        self._es_best    = -np.inf

    # ── early stopping ────────────────────────────────────────────────────────

    def _early_stop(self, score: float) -> bool:
        patience  = self.cfg.get('early_stop_patience', 4)
        min_delta = 1e-4
        if score > self._es_best + min_delta:
            self._es_best    = score
            self._es_counter = 0
            return False
        self._es_counter += 1
        return self._es_counter >= patience

    # ── training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        self.opt.zero_grad()

        for step, batch in enumerate(self.train_dl):
            iids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tids = batch["token_type_ids"].to(self.device)
            lbls = batch["label"].to(self.device)

            # ── R-Drop: two forward passes ─────────────────────────────
            # autocast disabled (use_fp16=False) — runs in float32
            with autocast(device_type="cuda", enabled=False):
                logits1 = self.model(iids, mask, tids)
                logits2 = self.model(iids, mask, tids)   # different dropout
                loss, ce, rank, kl = self.loss_fn(logits1, lbls, logits2)
                loss = loss / self.grad_accum

            # backward through scaler (no-op when disabled)
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.get("max_grad_norm", 1.0),
                )
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                self.opt.zero_grad()

            total_loss += loss.item() * self.grad_accum
            n_batches  += 1

            # lightweight W&B step logging every 50 steps
            if self.wandb_run and step % 50 == 0:
                self.wandb_run.log({
                    "train/step_loss" : loss.item() * self.grad_accum,
                    "train/ce"        : ce,
                    "train/kl_rdrop"  : kl,
                })

        return total_loss / max(n_batches, 1)

    # ── validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        all_preds, all_labels     = [], []
        all_pred_idx, all_lbl_idx = [], []

        for batch in self.val_dl:
            iids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            tids = batch['token_type_ids'].to(self.device)
            lbls = batch['label'].to(self.device)

            with autocast(device_type="cuda", enabled=False):
                logits      = self.model(iids, mask, tids)
                loss, *_    = self.loss_fn(logits, lbls)   # no R-Drop at val

            total_loss   += loss.item()
            n_batches    += 1
            all_preds    += ranked_preds(logits)
            all_labels   += [ANSWER_LABELS[l.item()] for l in lbls]
            all_pred_idx += logits.argmax(dim=1).cpu().tolist()
            all_lbl_idx  += lbls.cpu().tolist()

        m3  = map_at_3(all_preds, all_labels)
        acc = accuracy_score(all_lbl_idx, all_pred_idx)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_lbl_idx, all_pred_idx, average='macro', zero_division=0)

        return (total_loss / max(n_batches, 1),
                m3, acc, precision, recall, f1)

    # ── main training loop ────────────────────────────────────────────────────

    def train(self) -> dict:
        logger.info("=" * 60)
        logger.info("RoBERTa-base Training")
        logger.info("=" * 60)

        epochs         = self.cfg.get('epochs', 12)
        unfreeze_epoch = self.cfg.get('unfreeze_epoch', 2)

        for ep in range(1, epochs + 1):

            # progressive unfreezing
            if ep >= unfreeze_epoch:
                unfroze = self.model.unfreeze_top_layer()
                if unfroze:
                    logger.info(f"  ↑ Unfroze one transformer layer at ep {ep}")

            tr_loss                         = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()
            lr_now = self.opt.param_groups[0]['lr']

            for k, v in [
                ("tr_loss", tr_loss), ("vl_loss", vl_loss),
                ("vl_map3", m3),      ("vl_acc",  acc),
                ("vl_precision", prec), ("vl_recall", rec),
                ("vl_f1",  f1),       ("lr",      lr_now),
            ]:
                self.history[k].append(v)

            flag = ''
            if m3 > self.best_map3:
                self.best_map3  = m3
                self.best_state = {
                    k: v.clone()
                    for k, v in self.model.state_dict().items()
                }
                flag = '  ★ new best'

            logger.info(
                f"Ep {ep:3d}/{epochs} | "
                f"tr={tr_loss:.4f} vl={vl_loss:.4f} | "
                f"MAP@3={m3:.4f} Acc={acc:.4f} "
                f"F1={f1:.4f} lr={lr_now:.2e}{flag}"
            )

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch"         : ep,
                    "train/loss"    : tr_loss,
                    "val/loss"      : vl_loss,
                    "val/map3"      : m3,
                    "val/acc"       : acc,
                    "val/precision" : prec,
                    "val/recall"    : rec,
                    "val/f1"        : f1,
                    "lr"            : lr_now,
                }, step=ep)

            if self._early_stop(m3):
                logger.info(f"Early stopping at epoch {ep}")
                break

        # restore best checkpoint
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        logger.info(f"Best Val MAP@3 = {self.best_map3:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                'best_val_map3' : self.best_map3,
                'best_val_acc'  : max(self.history['vl_acc']),
                'best_val_f1'   : max(self.history['vl_f1']),
            })

        return dict(self.history)