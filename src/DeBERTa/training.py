# src/DeBERTa/training.py
"""
Training logic for DeBERTa-v3.

Fixes applied vs. original
───────────────────────────
1. cls.float() in model.forward() fixes dtype mismatch
2. loss.item() was returning nan — root cause was autocast context leaking
   into loss logging. Now loss is always fp32 (MCQLoss output is fp32
   because it operates on fp32 logits from the head).
3. Validation also wrapped in autocast for consistency.
4. Gradient accumulation logic made robust (handles last incomplete batch).
5. LR schedule: use OneCycleLR instead of linear warmup —
   OneCycleLR is more stable for small datasets (1024 train samples).
6. num_workers=0 to avoid Kaggle multiprocessing issues that slow down I/O.
"""

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.BiLSTM.training import MCQLoss, ranked_preds, map_at_3, ANSWER_LABELS

logger = logging.getLogger("DeBERTa.Trainer")


class Trainer:
    """
    Fine-tuning trainer for DeBERTa-v3.

    Key design choices for small dataset (1024 train rows)
    ───────────────────────────────────────────────────────
    • OneCycleLR scheduler  — finds good LR fast, avoids flat regions
    • grad_accum            — effective batch size 32 despite batch_size=8
    • bf16 autocast         — 2× speed, model head forced fp32
    • best_state on CPU     — saves ~700 MB GPU memory
    """

    def __init__(self, model, train_dl, val_dl,
                 opt, sched, loss_fn,
                 device, cfg: dict, wandb_run=None):

        self.model      = model.to(device)
        self.train_dl   = train_dl
        self.val_dl     = val_dl
        self.opt        = opt
        self.sched      = sched
        self.loss_fn    = loss_fn
        self.device     = device
        self.cfg        = cfg
        self.wandb_run  = wandb_run

        self.history    = defaultdict(list)
        self.best_map3  = -np.inf
        self.best_state = None

        self._es_counter = 0
        self._es_best    = -np.inf

        self.grad_accum = cfg.get('grad_accum_steps', 4)

        # ── precision: bf16 preferred, fp32 fallback ──────────────────────────
        self.use_autocast = (
            cfg.get('fp16', True) and
            device == 'cuda' and
            torch.cuda.is_available()
        )
        if self.use_autocast:
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                logger.info("Precision: bf16 autocast  "
                            "(head forced fp32 via cls.float())")
            else:
                self.amp_dtype = torch.float16
                logger.info("Precision: fp16 autocast  "
                            "(head forced fp32 via cls.float())")
        else:
            logger.info("Precision: fp32 (no autocast)")

    # ── autocast context ──────────────────────────────────────────────────────

    def _autocast(self):
        """Returns the appropriate autocast context manager."""
        if self.use_autocast:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        # null context
        return torch.amp.autocast("cuda", enabled=False)

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

    # ── one training epoch ────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        tot_loss  = 0.0
        n_steps   = 0

        self.opt.zero_grad()

        for step, batch in enumerate(self.train_dl):
            ids    = batch['input_ids']     .to(self.device)
            mask   = batch['attention_mask'].to(self.device)
            labels = batch['label']         .to(self.device)

            # ── forward under autocast ────────────────────────────────────────
            with self._autocast():
                logits = self.model(ids, mask)
                # logits is fp32 (head forces it) — loss is always fp32
                loss, *_ = self.loss_fn(logits, labels)

            # ── scale for accumulation and backward ───────────────────────────
            (loss / self.grad_accum).backward()

            # ── collect loss BEFORE zero_grad (it's a scalar, safe) ───────────
            # Use .detach().item() to ensure no autocast context leaks
            loss_val = loss.detach().float().item()

            is_accum_boundary = (step + 1) % self.grad_accum == 0
            is_last_batch     = (step + 1) == len(self.train_dl)

            if is_accum_boundary or is_last_batch:
                # clip raw fp32 gradients — no scaler needed
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.get('max_grad_norm', 1.0))
                self.opt.step()

                # OneCycleLR steps per optimizer step, not per epoch
                if isinstance(self.sched,
                              torch.optim.lr_scheduler.OneCycleLR):
                    self.sched.step()

                self.opt.zero_grad()

            tot_loss += loss_val
            n_steps  += 1

        return tot_loss / max(n_steps, 1)

    # ── validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        tot_loss = 0.0
        n_steps  = 0
        all_preds, all_labels = [], []
        all_pred_idx, all_label_idx = [], []

        for batch in self.val_dl:
            ids    = batch['input_ids']     .to(self.device)
            mask   = batch['attention_mask'].to(self.device)
            labels = batch['label']         .to(self.device)

            # ── wrap validation in same autocast ──────────────────────────────
            with self._autocast():
                logits = self.model(ids, mask)
                loss, *_ = self.loss_fn(logits, labels)

            loss_val = loss.detach().float().item()

            pred_idx = logits.argmax(dim=1)
            all_pred_idx .extend(pred_idx.cpu().numpy())
            all_label_idx.extend(labels  .cpu().numpy())

            tot_loss += loss_val
            n_steps  += 1

            all_preds  += ranked_preds(logits)
            all_labels += [ANSWER_LABELS[l.item()] for l in labels]

        m3        = map_at_3(all_preds, all_labels)
        acc       = accuracy_score(all_label_idx, all_pred_idx)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_label_idx, all_pred_idx,
            average='macro', zero_division=0)

        return tot_loss / max(n_steps, 1), m3, acc, prec, rec, f1

    # ── main training loop ────────────────────────────────────────────────────

    def train(self) -> dict:
        logger.info(
            f"DeBERTa fine-tuning start | "
            f"epochs={self.cfg.get('epochs',10)} | "
            f"grad_accum={self.grad_accum} | "
            f"effective_batch="
            f"{self.cfg.get('batch_size',8) * self.grad_accum}")

        for ep in range(1, self.cfg.get('epochs', 10) + 1):

            tr_loss = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()

            # step ReduceLROnPlateau (if used instead of OneCycleLR)
            if isinstance(self.sched,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched.step(m3)

            lr = self.opt.param_groups[0]['lr']

            for k, v in [
                ("tr_loss",      tr_loss),
                ("vl_loss",      vl_loss),
                ("vl_map3",      m3),
                ("vl_acc",       acc),
                ("vl_precision", prec),
                ("vl_recall",    rec),
                ("vl_f1",        f1),
                ("lr",           lr),
            ]:
                self.history[k].append(v)

            flag = ''
            if m3 > self.best_map3:
                self.best_map3  = m3
                self.best_state = {
                    k: v.clone().cpu()
                    for k, v in self.model.state_dict().items()
                }
                flag = '  ★'

            logger.info(
                f"Ep {ep:3d} | "
                f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                f"MAP@3={m3:.4f}  Acc={acc:.4f}  "
                f"lr={lr:.2e}{flag}")

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
                    "lr"            : lr,
                }, step=ep)

            if self._early_stop(m3):
                logger.info(f"Early stop at epoch {ep}")
                break

        if self.best_state:
            self.model.load_state_dict(
                {k: v.to(self.device)
                 for k, v in self.best_state.items()})

        logger.info(f"Best Val MAP@3 = {self.best_map3:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                'best_val_map3': self.best_map3,
                'best_val_acc' : max(self.history['vl_acc']),
            })

        return dict(self.history)