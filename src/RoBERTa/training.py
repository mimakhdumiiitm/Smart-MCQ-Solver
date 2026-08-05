# src/RoBERTa/training.py
"""
Training logic for RoBERTa-base MCQ.

Key differences vs DeBERTa trainer
─────────────────────────────────────
  - AMP: float32 parameters + autocast(dtype=torch.float16) only on forward
    → GradScaler stays in float32 mode: NO "Attempting to unscale FP16" error
  - SWA (Stochastic Weight Averaging) for better final weights
  - All other features identical: grad accum, progressive unfreeze,
    early stopping, per-step W&B logging
"""

import logging
import math
from collections import defaultdict
from copy import deepcopy

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
    CE (with label smoothing)  +  pairwise margin ranking loss.

    Both terms are computed in float32 regardless of AMP context
    because logits are already cast back before loss computation.
    """

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

    def forward(
        self,
        logits  : torch.Tensor,   # [B, 5]  float32
        targets : torch.Tensor,   # [B]     long
    ):
        # ensure float32 (safe even inside autocast block)
        logits = logits.float()

        ce_loss  = self.ce(logits, targets)

        # pairwise: push correct option score > all others by margin
        pos      = logits.gather(1, targets.unsqueeze(1))          # [B, 1]
        margin   = F.relu(self.margin - (pos - logits))            # [B, 5]
        eye      = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.)
        rank_loss = (margin * (1 - eye)).sum(1).mean()

        total = self.ce_w * ce_loss + self.rank_w * rank_loss
        return total, ce_loss.item(), rank_loss.item()


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
# Simple SWA helper
# ─────────────────────────────────────────────────────────────────────────────

class SWAHelper:
    """
    Maintain a running average of model weights (Stochastic Weight Averaging).
    Cheaper than torch.optim.swa_utils and works with DataParallel.

    Usage
    ─────
        swa = SWAHelper(model)
        # at end of each qualifying epoch:
        swa.update(model)
        # at end of training:
        swa.apply(model)   # set model weights to SWA average
    """

    def __init__(self, model: nn.Module):
        # store a CPU copy so we don't waste GPU memory
        self._avg   = deepcopy(model).cpu()
        self._n     = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._n += 1
        alpha = 1.0 / self._n
        for p_avg, p_cur in zip(
            self._avg.parameters(),
            model.parameters(),
        ):
            # incremental mean: avg = avg + (new - avg) / n
            p_avg.data.add_(
                (p_cur.detach().float().cpu() - p_avg.data) * alpha
            )

    @torch.no_grad()
    def apply(self, model: nn.Module):
        """Copy averaged weights back to model (on its current device)."""
        device = next(model.parameters()).device
        for p_avg, p_cur in zip(
            self._avg.parameters(),
            model.parameters(),
        ):
            p_cur.data.copy_(p_avg.data.to(device))
        logger.info(f"SWA applied (n={self._n} snapshots).")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    RoBERTa MCQ Trainer.

    AMP Strategy (avoids "unscale FP16 gradients" error)
    ──────────────────────────────────────────────────────
    • Model parameters stay float32 at all times.
    • autocast(device_type="cuda", dtype=torch.float16) wraps only the
      encoder forward pass, producing float16 activations for speed.
    • GradScaler therefore always sees float32 gradients — safe.

    Features
    ────────
    ✓  AMP mixed precision (float32 params, float16 activations)
    ✓  GradScaler (no FP16 gradient error)
    ✓  Gradient accumulation
    ✓  Progressive layer unfreezing
    ✓  SWA
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
        cfg       : dict,
        device    : str,
        wandb_run  = None,
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

        # ── AMP setup ─────────────────────────────────────────────────────────
        # use_fp16 = True  → float16 activations, float32 params + grads
        self.use_amp = cfg.get('use_fp16', True) and device != 'cpu'
        # GradScaler in float32 mode — safe, no FP16 gradient issue
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        self.grad_accum = cfg.get('grad_accum', 4)

        # ── SWA ───────────────────────────────────────────────────────────────
        self.use_swa        = cfg.get('use_swa', True)
        self.swa_start_ep   = cfg.get('swa_start_epoch', 7)
        self.swa_lr         = cfg.get('swa_lr', 5e-6)
        self._swa           = SWAHelper(model) if self.use_swa else None

        # ── state ─────────────────────────────────────────────────────────────
        self.history    = defaultdict(list)
        self.best_map3  = -np.inf
        self.best_state = None

        self._es_counter = 0
        self._es_best    = -np.inf

    # ── early stopping ────────────────────────────────────────────────────────

    def _early_stop(self, score: float) -> bool:
        patience  = self.cfg.get('early_stop_patience', 5)
        min_delta = 1e-4
        if score > self._es_best + min_delta:
            self._es_best    = score
            self._es_counter = 0
            return False
        self._es_counter += 1
        return self._es_counter >= patience

    # ── SWA lr switch ─────────────────────────────────────────────────────────

    def _set_swa_lr(self):
        """Lower all param-group LRs to swa_lr."""
        for pg in self.opt.param_groups:
            pg['lr'] = self.swa_lr
        logger.info(f"SWA phase: lr set to {self.swa_lr:.2e} for all groups.")

    # ── one training epoch ────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        self.opt.zero_grad()

        for step, batch in enumerate(self.train_dl):
            iids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            # token_type_ids are zeros; pass to model (it ignores them)
            tids = batch["token_type_ids"].to(self.device)
            lbls = batch["label"].to(self.device)

            # ── forward under AMP ─────────────────────────────────────────────
            # autocast produces float16 activations → faster matmuls on GPU
            # model parameters remain float32 → GradScaler safe
            with autocast(device_type="cuda", dtype=torch.float16,
                          enabled=self.use_amp):
                logits = self.model(iids, mask, tids)

            # loss computed in float32 (logits.float() inside MCQLoss)
            loss, ce, rank = self.loss_fn(logits, lbls)
            loss = loss / self.grad_accum

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

        # handle leftover steps (when len(train_dl) % grad_accum != 0)
        remaining = len(self.train_dl) % self.grad_accum
        if remaining != 0:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.get("max_grad_norm", 1.0),
            )
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

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

            with autocast(device_type="cuda", dtype=torch.float16,
                          enabled=self.use_amp):
                logits = self.model(iids, mask, tids)

            loss, *_ = self.loss_fn(logits, lbls)

            total_loss   += loss.item()
            n_batches    += 1
            all_preds    += ranked_preds(logits.float())
            all_labels   += [ANSWER_LABELS[l.item()] for l in lbls]
            all_pred_idx += logits.float().argmax(dim=1).cpu().tolist()
            all_lbl_idx  += lbls.cpu().tolist()

        m3  = map_at_3(all_preds, all_labels)
        acc = accuracy_score(all_lbl_idx, all_pred_idx)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_lbl_idx, all_pred_idx,
            average='macro', zero_division=0,
        )
        return (total_loss / max(n_batches, 1),
                m3, acc, precision, recall, f1)

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self) -> dict:
        logger.info("=" * 60)
        logger.info("RoBERTa-base Training")
        logger.info("=" * 60)

        epochs         = self.cfg.get('epochs', 12)
        unfreeze_epoch = self.cfg.get('unfreeze_epoch', 2)
        swa_active     = False

        for ep in range(1, epochs + 1):

            # ── progressive unfreezing ────────────────────────────────────────
            if ep >= unfreeze_epoch:
                unfroze = self.model.unfreeze_top_layer()
                if unfroze:
                    logger.info(f"  ↑ Unfroze one transformer layer at ep {ep}")

            # ── SWA phase transition ──────────────────────────────────────────
            if self.use_swa and ep == self.swa_start_ep and not swa_active:
                self._set_swa_lr()
                swa_active = True
                logger.info(f"SWA activated at epoch {ep}")

            tr_loss                         = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()
            lr_now = self.opt.param_groups[0]['lr']

            # ── SWA update ────────────────────────────────────────────────────
            if swa_active and self._swa is not None:
                self._swa.update(self.model)

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
                    "epoch"          : ep,
                    "train/loss"     : tr_loss,
                    "val/loss"       : vl_loss,
                    "val/map3"       : m3,
                    "val/acc"        : acc,
                    "val/precision"  : prec,
                    "val/recall"     : rec,
                    "val/f1"         : f1,
                    "lr"             : lr_now,
                    "swa_active"     : int(swa_active),
                }, step=ep)

            if self._early_stop(m3):
                logger.info(f"Early stopping at epoch {ep}")
                break

        # ── finalise weights ──────────────────────────────────────────────────
        if self.use_swa and self._swa is not None and self._swa._n > 0:
            # apply SWA weights and re-validate
            self._swa.apply(self.model)
            logger.info("Re-validating with SWA weights …")
            _, swa_m3, swa_acc, *_ = self._validate()
            logger.info(f"SWA MAP@3={swa_m3:.4f}  Acc={swa_acc:.4f}")

            if swa_m3 >= self.best_map3:
                self.best_map3 = swa_m3
                logger.info("SWA weights are best — keeping.")
            else:
                # revert to best checkpoint
                logger.info(
                    f"SWA ({swa_m3:.4f}) < best ({self.best_map3:.4f}) "
                    f"— reverting to checkpoint."
                )
                if self.best_state:
                    self.model.load_state_dict(self.best_state)
        elif self.best_state:
            self.model.load_state_dict(self.best_state)

        logger.info(f"Best Val MAP@3 = {self.best_map3:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                'best_val_map3' : self.best_map3,
                'best_val_acc'  : max(self.history['vl_acc']),
                'best_val_f1'   : max(self.history['vl_f1']),
            })

        return dict(self.history)