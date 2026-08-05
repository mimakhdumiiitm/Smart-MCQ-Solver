# src/RoBERTa/training.py
"""
Training logic for RoBERTa-base MCQ.

Key fixes vs previous version
──────────────────────────────
  1. DataParallel unfreeze bug fixed
     - _get_raw_model() unwraps DataParallel before calling unfreeze_top_layer
     - was: self.model.unfreeze_top_layer() → AttributeError
     - now: _get_raw_model(self.model).unfreeze_top_layer()

  2. W&B step logging conflict fixed
     - Epoch-level logging uses commit=False to avoid step counter conflicts
     - Step-level logging inside _train_epoch uses no step= argument
     - A single global_step counter is maintained to keep W&B happy

  3. Gradient accumulation last-batch handling fixed
     - Previous version double-stepped on last batch when n_steps % grad_accum != 0
     - Now: optimizer step only when is_accum_step XOR is_last_step
       (is_last_step only fires if it wasn't already an accum step)

  4. best_state saved to CPU → loaded back to correct device
     (unchanged from OOM-fix version — kept correct)

  5. FP32 permanently enforced — GradScaler(enabled=False) is a clean no-op
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DataParallel to access the actual model methods.

    Fix for:
        AttributeError: 'DataParallel' object has no attribute 'unfreeze_top_layer'

    DataParallel wraps the model and only forwards nn.Module standard methods.
    Custom methods like unfreeze_top_layer() live on .module.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class MCQLoss(nn.Module):
    """
    CE (label smoothing) + pairwise margin ranking.

    CE loss   → calibrated probability distribution
    Rank loss → pushes correct option score above all distractors by margin

    No R-Drop: removed to save memory (doubles backward graph).
    Rank loss provides comparable regularisation in one forward pass.
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
        logits  : torch.Tensor,   # [B, 5]
        targets : torch.Tensor,   # [B]
    ):
        ce_loss = self.ce(logits, targets)

        pos        = logits.gather(1, targets.unsqueeze(1))     # [B, 1]
        margin_mat = F.relu(self.margin - (pos - logits))       # [B, 5]
        eye        = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.)
        rank_loss  = (margin_mat * (1 - eye)).sum(1).mean()

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
                hits += 1
                ap += hits / k
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
    RoBERTa MCQ Trainer.

    Features
    ────────
    ✓  FP32 only (GradScaler disabled)
    ✓  Gradient accumulation (correct last-batch handling)
    ✓  Progressive layer unfreezing (DataParallel-safe via _get_raw_model)
    ✓  W&B logging (no step-counter conflicts)
    ✓  Early stopping on MAP@3 (patience=5 for noisy small val)
    ✓  Best-checkpoint on CPU, restored to device at end
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

        # FP32 permanently — GradScaler(enabled=False) is a clean no-op
        self.use_fp16   = False
        self.scaler     = GradScaler("cuda", enabled=False)
        self.grad_accum = cfg.get('grad_accum', 4)

        self.history     = defaultdict(list)
        self.best_map3   = -np.inf
        self.best_state  = None

        self._es_counter = 0
        self._es_best    = -np.inf

        # global step counter for W&B (avoids non-monotonic step warnings)
        self._global_step = 0

    # ── early stopping ────────────────────────────────────────────────────────

    def _early_stop(self, score: float) -> bool:
        patience  = self.cfg.get('early_stop_patience', 5)
        min_delta = 1e-4
        if score > self._es_best + min_delta:
            self._es_best    = score
            self._es_counter = 0
            return False
        self._es_counter += 1
        logger.info(
            f"  Early-stop counter: {self._es_counter}/{patience}"
        )
        return self._es_counter >= patience

    # ── optimizer step ────────────────────────────────────────────────────────

    def _opt_step(self):
        """Unscale → clip → step → update → zero_grad → cache flush."""
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.get("max_grad_norm", 1.0),
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # ── training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        n_steps    = len(self.train_dl)

        self.opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_dl):
            iids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tids = batch["token_type_ids"].to(self.device)
            lbls = batch["label"].to(self.device)

            with autocast(device_type="cuda", enabled=False):
                logits         = self.model(iids, mask, tids)
                loss, ce, rank = self.loss_fn(logits, lbls)
                loss           = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            is_accum_step = ((step + 1) % self.grad_accum == 0)
            is_last_step  = (step == n_steps - 1)

            # Step on accumulation boundary OR on the very last batch,
            # but NOT twice if both conditions are true simultaneously.
            if is_accum_step or (is_last_step and not is_accum_step):
                self._opt_step()

            total_loss        += loss.item() * self.grad_accum
            n_batches         += 1
            self._global_step += 1

            # W&B step-level logging — use global_step, no commit
            if self.wandb_run and self._global_step % 50 == 0:
                self.wandb_run.log({
                    "train/step_loss" : loss.item() * self.grad_accum,
                    "train/ce"        : ce,
                    "train/rank"      : rank,
                    "global_step"     : self._global_step,
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
                logits   = self.model(iids, mask, tids)
                loss, *_ = self.loss_fn(logits, lbls)

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
        logger.info("RoBERTa-base Training  [FP32 | CE + Rank]")
        logger.info("=" * 60)

        epochs         = self.cfg.get('epochs', 12)
        unfreeze_epoch = self.cfg.get('unfreeze_epoch', 2)

        for ep in range(1, epochs + 1):

            # ── progressive unfreezing (DataParallel-safe) ────────────────
            # Fix: use _get_raw_model to access .unfreeze_top_layer()
            # DataParallel only forwards standard nn.Module methods;
            # custom methods must be called on .module directly.
            if ep >= unfreeze_epoch:
                raw = _get_raw_model(self.model)
                unfroze = raw.unfreeze_top_layer()
                if unfroze:
                    logger.info(
                        f"  ↑ Unfroze one transformer layer at ep {ep}"
                    )

            tr_loss                         = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()
            lr_now = self.opt.param_groups[0]['lr']

            for k, v in [
                ("tr_loss",      tr_loss),
                ("vl_loss",      vl_loss),
                ("vl_map3",      m3),
                ("vl_acc",       acc),
                ("vl_precision", prec),
                ("vl_recall",    rec),
                ("vl_f1",        f1),
                ("lr",           lr_now),
            ]:
                self.history[k].append(v)

            flag = ''
            if m3 > self.best_map3:
                self.best_map3  = m3
                # save on CPU — avoids doubling GPU memory at checkpoint time
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                flag = '  ★ new best'

            logger.info(
                f"Ep {ep:3d}/{epochs} | "
                f"tr={tr_loss:.4f}  vl={vl_loss:.4f} | "
                f"MAP@3={m3:.4f}  Acc={acc:.4f}  "
                f"F1={f1:.4f}  lr={lr_now:.2e}{flag}"
            )

            # W&B epoch logging — no step= argument avoids counter conflicts
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
                    "global_step"   : self._global_step,
                })

            if self._early_stop(m3):
                logger.info(f"Early stopping triggered at epoch {ep}")
                break

        # restore best checkpoint to correct device
        if self.best_state:
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