# src/RoBERTa/training.py
"""
Training logic for RoBERTa-base MCQ.

Changes from reviewed version
──────────────────────────────
  Critical fixes
  ─────────────
  1. Unfrozen params added to optimizer (was: unfrozen params got
     gradients but no optimizer update — silent accuracy killer).
     _add_unfrozen_params_to_optimizer() handles this correctly.

  2. best_state saved from the UNWRAPPED model (no "module." key prefix).
     Restoring to the unwrapped model therefore works without key-name
     surgery, whether or not DataParallel is in use.

  Medium fixes
  ────────────
  3. autocast is conditional on self.use_fp16 (was always enabled=False,
     which was a functional no-op but misleading and added call overhead).
     FP32 is still the default; set use_fp16=True in config to enable.

  4. Early stopping now has a grace period (early_stop_grace epochs) so
     the ES counter never fires while the model is still warming up and
     most layers are frozen.

  Low fixes
  ─────────
  5. ranked_preds: single CPU transfer via .cpu().numpy() instead of
     per-element .item() calls.

  Other
  ─────
  6. _get_raw_model() unchanged — still the DataParallel unwrap helper.
  7. Gradient-accumulation last-batch logic unchanged — was already correct.
  8. W&B logging unchanged — no step-counter conflicts.
"""

import contextlib
import logging
import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import GradScaler, autocast

logger = logging.getLogger("RoBERTa.Trainer")

ANSWER_LABELS = ["A", "B", "C", "D", "E"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_model(model: nn.Module) -> nn.Module:
    """
    Return the unwrapped model even if it is inside nn.DataParallel.

    DataParallel only forwards standard nn.Module methods; custom methods
    (unfreeze_top_layer, etc.) must be called on .module directly.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class MCQLoss(nn.Module):
    """
    Cross-entropy (label-smoothed) + pairwise margin ranking.

    CE loss   → well-calibrated probability distribution
    Rank loss → pushes the correct option score above every distractor
                by at least *margin*
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
        ce_loss    = self.ce(logits, targets)

        pos        = logits.gather(1, targets.unsqueeze(1))     # [B, 1]
        margin_mat = F.relu(self.margin - (pos - logits))       # [B, 5]
        eye        = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        rank_loss  = (margin_mat * (1.0 - eye)).sum(1).mean()

        total = self.ce_w * ce_loss + self.rank_w * rank_loss
        return total, ce_loss.item(), rank_loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ranked_preds(logits: torch.Tensor):
    """
    Return top-3 predicted labels for each row.

    Single GPU→CPU transfer via .cpu().numpy() rather than per-element
    .item() calls.
    """
    top3_idx = (
        torch.argsort(logits, dim=-1, descending=True)[:, :3]
        .cpu()
        .numpy()
    )
    return [[ANSWER_LABELS[int(i)] for i in row] for row in top3_idx]


def map_at_3(preds, labels) -> float:
    scores = []
    for pred, gold in zip(preds, labels):
        ap, hits = 0.0, 0
        for k, p in enumerate(pred[:3], 1):
            if p == gold:
                hits += 1
                ap   += hits / k
        scores.append(ap)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, n_warmup: int, n_total: int):
    """Linear warmup → cosine decay."""
    def lr_lambda(step: int) -> float:
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

    Key properties
    ──────────────
    ✓ FP32 by default; optional FP16 via cfg['use_fp16']=True
    ✓ Gradient accumulation with correct last-batch handling
    ✓ Progressive layer unfreezing (DataParallel-safe)
    ✓ Unfrozen params automatically added to optimizer param groups
    ✓ best_state saved from / restored to the UNWRAPPED model
      (no "module." key-prefix mismatch)
    ✓ Early stopping with configurable grace period
    ✓ W&B logging without step-counter conflicts
    """

    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        optimizer,
        scheduler,
        loss_fn    : MCQLoss,
        cfg        : Dict,
        device     : str,
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

        self.use_fp16   = bool(cfg.get("use_fp16", False))
        self.grad_accum = int(cfg.get("grad_accum", 4))

        # GradScaler is a no-op when enabled=False (FP32 mode)
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        self.history    : Dict           = defaultdict(list)
        self.best_map3  : float          = -np.inf
        self.best_state : Optional[Dict] = None

        self._es_counter    : int   = 0
        self._es_best       : float = -np.inf
        self._global_step   : int   = 0

        # autocast context: active for FP16, identity context for FP32
        self._autocast = (
            lambda: autocast(device_type="cuda", enabled=True)
            if self.use_fp16
            else contextlib.nullcontext
        )()

    # ── early stopping ────────────────────────────────────────────────────────

    def _early_stop(self, score: float, epoch: int) -> bool:
        """
        Return True when training should stop.

        The ES counter is not incremented during the grace period
        (first *early_stop_grace* epochs) so that the model has time
        to warm up before we start watching for stagnation.
        """
        patience    = int(self.cfg.get("early_stop_patience", 5))
        grace       = int(self.cfg.get("early_stop_grace",   3))
        min_delta   = 1e-4

        if epoch <= grace:
            return False    # grace period — never trigger

        if score > self._es_best + min_delta:
            self._es_best    = score
            self._es_counter = 0
            return False

        self._es_counter += 1
        logger.info(
            "  Early-stop counter: %d/%d", self._es_counter, patience
        )
        return self._es_counter >= patience

    # ── optimizer step ────────────────────────────────────────────────────────

    def _opt_step(self) -> None:
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            float(self.cfg.get("max_grad_norm", 1.0)),
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ── unfreezing + optimizer sync ──────────────────────────────────────────

    def _unfreeze_and_sync(self, epoch: int) -> None:
        """
        Unfreeze the topmost still-frozen transformer layer and immediately
        register the newly trainable parameters with the optimizer.

        This is critical: without the param-group update the unfrozen
        parameters receive gradient signal but are never updated by the
        optimizer (silent accuracy regression).
        """
        raw     = _get_raw_model(self.model)
        unfroze = raw.unfreeze_top_layer()
        if not unfroze:
            return

        logger.info("  ↑ Unfroze one transformer layer at epoch %d", epoch)

        # Collect IDs of parameters already known to the optimizer
        existing_ids = {
            id(p)
            for group in self.opt.param_groups
            for p in group["params"]
        }

        # Find newly trainable parameters not yet in any group
        new_params = [
            p
            for p in raw.encoder.parameters()
            if p.requires_grad and id(p) not in existing_ids
        ]

        if new_params:
            lr_bb = float(self.cfg.get("lr_backbone", 2e-5))
            wd    = float(self.cfg.get("weight_decay", 0.01))
            self.opt.add_param_group(
                {"params": new_params, "lr": lr_bb, "weight_decay": wd}
            )
            logger.info(
                "  Added %d newly unfrozen params to optimizer "
                "(lr=%.2e, wd=%.4f)",
                len(new_params), lr_bb, wd,
            )

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

            with self._autocast:
                logits         = self.model(iids, mask, tids)
                loss, ce, rank = self.loss_fn(logits, lbls)
                loss           = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            is_accum_step = ((step + 1) % self.grad_accum == 0)
            is_last_step  = (step == n_steps - 1)

            if is_accum_step or (is_last_step and not is_accum_step):
                self._opt_step()

            total_loss        += loss.item() * self.grad_accum
            n_batches         += 1
            self._global_step += 1

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
        total_loss  = 0.0
        n_batches   = 0
        all_preds   : list = []
        all_labels  : list = []
        all_pred_idx: list = []
        all_lbl_idx : list = []

        for batch in self.val_dl:
            iids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tids = batch["token_type_ids"].to(self.device)
            lbls = batch["label"].to(self.device)

            with self._autocast:
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
            all_lbl_idx, all_pred_idx,
            average      = "macro",
            zero_division = 0,
        )

        return (
            total_loss / max(n_batches, 1),
            m3, acc, precision, recall, f1,
        )

    # ── save / restore best state ─────────────────────────────────────────────

    def _save_best(self) -> None:
        """
        Save model weights to CPU from the UNWRAPPED model.

        Keys have no "module." prefix, so the checkpoint can be loaded
        into either a plain MCQRoBERTa or an unwrapped DataParallel model
        without key-name surgery.
        """
        raw = _get_raw_model(self.model)
        self.best_state = {
            k: v.cpu().clone()
            for k, v in raw.state_dict().items()
        }

    def _restore_best(self) -> None:
        """Restore the best checkpoint to the unwrapped model on device."""
        if self.best_state is None:
            return
        raw = _get_raw_model(self.model)
        raw.load_state_dict(
            {k: v.to(self.device) for k, v in self.best_state.items()}
        )

    # ── main training loop ────────────────────────────────────────────────────

    def train(self) -> Dict:
        logger.info("=" * 60)
        logger.info(
            "RoBERTa-base Training  [%s | CE + Rank]",
            "FP16" if self.use_fp16 else "FP32",
        )
        logger.info("=" * 60)

        epochs         = int(self.cfg.get("epochs",         12))
        unfreeze_epoch = int(self.cfg.get("unfreeze_epoch",  2))

        for ep in range(1, epochs + 1):

            # ── progressive unfreezing (DataParallel-safe) ─────────────
            if ep >= unfreeze_epoch:
                self._unfreeze_and_sync(ep)

            tr_loss                         = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()
            lr_now = self.opt.param_groups[0]["lr"]

            for key, val in [
                ("tr_loss",      tr_loss),
                ("vl_loss",      vl_loss),
                ("vl_map3",      m3),
                ("vl_acc",       acc),
                ("vl_precision", prec),
                ("vl_recall",    rec),
                ("vl_f1",        f1),
                ("lr",           lr_now),
            ]:
                self.history[key].append(val)

            flag = ""
            if m3 > self.best_map3:
                self.best_map3 = m3
                self._save_best()
                flag = "  ★ new best"

            logger.info(
                "Ep %3d/%d | tr=%.4f  vl=%.4f | "
                "MAP@3=%.4f  Acc=%.4f  F1=%.4f  lr=%.2e%s",
                ep, epochs, tr_loss, vl_loss, m3, acc, f1, lr_now, flag,
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
                    "global_step"   : self._global_step,
                })

            if self._early_stop(m3, ep):
                logger.info("Early stopping triggered at epoch %d.", ep)
                break

        self._restore_best()
        logger.info("Best Val MAP@3 = %.4f", self.best_map3)

        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                "best_val_map3" : self.best_map3,
                "best_val_acc"  : max(self.history["vl_acc"]),
                "best_val_f1"   : max(self.history["vl_f1"]),
            })

        return dict(self.history)