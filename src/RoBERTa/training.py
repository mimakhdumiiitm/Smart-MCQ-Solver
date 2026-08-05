# src/RoBERTa/training.py
"""
RoBERTa-base MCQ Trainer — memory-safe AMP, no FP16 gradient error.

AMP strategy
────────────
  • Model params stay float32.
  • autocast(device_type="cuda", dtype=torch.float16) wraps forward only.
  • GradScaler sees float32 grads → safe, no "unscale FP16" error.
  • Loss computed in float32 (logits.float() guard in MCQLoss).
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
    """CE (label-smoothed) + pairwise margin ranking."""

    def __init__(
        self,
        smoothing : float = 0.05,
        margin    : float = 0.3,
        ce_w      : float = 0.65,
        rank_w    : float = 0.35,
    ):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.margin = margin
        self.ce_w   = ce_w
        self.rank_w = rank_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits.float()                        # safe inside autocast
        ce_l   = self.ce(logits, targets)
        pos    = logits.gather(1, targets.unsqueeze(1))
        mrg    = F.relu(self.margin - (pos - logits))
        eye    = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.)
        rank_l = (mrg * (1 - eye)).sum(1).mean()
        total  = self.ce_w * ce_l + self.rank_w * rank_l
        return total, ce_l.item(), rank_l.item()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ranked_preds(logits: torch.Tensor):
    top3 = torch.argsort(logits.float(), dim=-1, descending=True)[:, :3]
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
# Scheduler
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
    Memory-efficient RoBERTa MCQ Trainer.

    Key safety measures
    ───────────────────
    1. autocast dtype=torch.float16 wraps ONLY the model forward.
    2. loss.float() guard in MCQLoss → GradScaler gets float32 grads.
    3. No DataParallel inside Trainer — handled in pipeline.
    4. Gradient accumulation with correct leftover handling.
    5. torch.cuda.empty_cache() called between train & validate.
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

        self.use_amp    = cfg.get('use_fp16', True) and device != 'cpu'
        # GradScaler in float32 mode — no FP16 parameter casting
        self.scaler     = GradScaler("cuda", enabled=self.use_amp)
        self.grad_accum = cfg.get('grad_accum', 8)

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

        n = len(self.train_dl)

        for step, batch in enumerate(self.train_dl):
            iids = batch["input_ids"].to(self.device,  non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)
            tids = batch["token_type_ids"].to(self.device, non_blocking=True)
            lbls = batch["label"].to(self.device,      non_blocking=True)

            # forward in float16 activations, float32 params
            with autocast(device_type="cuda", dtype=torch.float16,
                          enabled=self.use_amp):
                logits = self.model(iids, mask, tids)

            # loss always in float32 (MCQLoss casts logits)
            loss, _ce, _rk = self.loss_fn(logits, lbls)
            loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            is_last = (step == n - 1)
            if (step + 1) % self.grad_accum == 0 or is_last:
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

        return total_loss / max(n_batches, 1)

    # ── validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> tuple:
        self.model.eval()
        torch.cuda.empty_cache()

        total_loss = 0.0
        n_batches  = 0
        all_preds, all_labels     = [], []
        all_pred_idx, all_lbl_idx = [], []

        for batch in self.val_dl:
            iids = batch['input_ids'].to(self.device,     non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)
            tids = batch['token_type_ids'].to(self.device, non_blocking=True)
            lbls = batch['label'].to(self.device,          non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16,
                          enabled=self.use_amp):
                logits = self.model(iids, mask, tids)

            loss, *_ = self.loss_fn(logits, lbls)
            total_loss   += loss.item()
            n_batches    += 1

            lf = logits.float()
            all_preds    += ranked_preds(lf)
            all_labels   += [ANSWER_LABELS[l.item()] for l in lbls]
            all_pred_idx += lf.argmax(dim=1).cpu().tolist()
            all_lbl_idx  += lbls.cpu().tolist()

        torch.cuda.empty_cache()

        m3  = map_at_3(all_preds, all_labels)
        acc = accuracy_score(all_lbl_idx, all_pred_idx)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_lbl_idx, all_pred_idx, average='macro', zero_division=0)

        return (total_loss / max(n_batches, 1), m3, acc, prec, rec, f1)

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self) -> dict:
        logger.info("=" * 60)
        logger.info("RoBERTa-base Training  (float32 params + float16 AMP)")
        logger.info("=" * 60)

        epochs         = self.cfg.get('epochs', 10)
        unfreeze_epoch = self.cfg.get('unfreeze_epoch', 2)

        for ep in range(1, epochs + 1):

            # progressive unfreezing
            if ep >= unfreeze_epoch:
                raw = (self.model.module
                       if hasattr(self.model, 'module') else self.model)
                unfroze = raw.unfreeze_top_layer()
                if unfroze:
                    logger.info(f"  ↑ Unfroze one layer at ep {ep}")

            torch.cuda.empty_cache()
            tr_loss                         = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()
            lr_now = self.opt.param_groups[0]['lr']

            for k, v in [
                ("tr_loss", tr_loss), ("vl_loss", vl_loss),
                ("vl_map3", m3),      ("vl_acc",  acc),
                ("vl_precision", prec), ("vl_recall", rec),
                ("vl_f1", f1),        ("lr", lr_now),
            ]:
                self.history[k].append(v)

            flag = ''
            if m3 > self.best_map3:
                self.best_map3  = m3
                # save CPU copy to free GPU memory
                self.best_state = {
                    k: v.detach().cpu().clone()
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
                logger.info(f"Early stopping at epoch {ep}.")
                break

        # restore best weights
        if self.best_state:
            # map to current device
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )

        logger.info(f"Best Val MAP@3 = {self.best_map3:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                'best_val_map3' : self.best_map3,
                'best_val_acc'  : max(self.history['vl_acc']),
                'best_val_f1'   : max(self.history['vl_f1']),
            })

        return dict(self.history)