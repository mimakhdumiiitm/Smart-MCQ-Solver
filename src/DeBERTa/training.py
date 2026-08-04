# src/DeBERTa/training.py
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup

from src.BiLSTM.training import MCQLoss, ranked_preds, map_at_3, ANSWER_LABELS

logger = logging.getLogger("DeBERTa.Trainer")


class Trainer:
    """
    Fine-tuning trainer for DeBERTa-v3.

    Precision strategy
    ──────────────────
    fp16=True  →  tries bf16 first (no scaler needed, wider range)
                  falls back to fp16 WITHOUT GradScaler
                  (DeBERTa-v3 has mixed-dtype params that break unscale_)
    fp16=False →  full fp32
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

        # early stopping
        self._es_counter = 0
        self._es_best    = -np.inf

        self.grad_accum = cfg.get('grad_accum_steps', 4)

        # ── precision setup ───────────────────────────────────────────────────
        self.autocast_ctx, self.scaler = self._setup_precision(
            cfg.get('fp16', True), device)

    # ── precision factory ─────────────────────────────────────────────────────

    def _setup_precision(self, fp16_requested: bool, device: str):
        """
        Returns (autocast_context_manager, scaler_or_None).

        Priority
        ────────
        1. bf16  — no scaler, wide dynamic range, works with DeBERTa-v3
        2. fp16  — no scaler (DeBERTa-v3 mixed dtypes break GradScaler)
        3. fp32  — plain training
        """
        if not fp16_requested or device != "cuda":
            logger.info("Precision: fp32 (no autocast)")
            # null context — does nothing
            return torch.amp.autocast("cuda", enabled=False), None

        # ── prefer bf16 ───────────────────────────────────────────────────────
        if torch.cuda.is_bf16_supported():
            logger.info("Precision: bf16 autocast (no GradScaler needed)")
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            return ctx, None       # bf16 never needs a scaler

        # ── fall back to fp16 without scaler ──────────────────────────────────
        # DeBERTa-v3 keeps some params in fp32 (SentencePiece embeddings)
        # → GradScaler.unscale_() crashes on mixed dtypes
        # → run fp16 autocast for speed but skip loss scaling
        logger.warning(
            "bf16 not supported on this GPU — using fp16 autocast "
            "WITHOUT GradScaler (safe for DeBERTa-v3 mixed-dtype params). "
            "Loss may underflow on very long training runs; "
            "switch to a GPU with bf16 support (Ampere+) for best results."
        )
        ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        return ctx, None           # no scaler — avoids the unscale_ crash

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
        n_batches = 0

        self.opt.zero_grad()

        for step, batch in enumerate(self.train_dl):
            ids    = batch['input_ids']     .to(self.device)
            mask   = batch['attention_mask'].to(self.device)
            labels = batch['label']         .to(self.device)

            # ── forward ───────────────────────────────────────────────────────
            with self.autocast_ctx:
                logits   = self.model(ids, mask)
                loss, *_ = self.loss_fn(logits, labels)
                # scale for gradient accumulation
                loss_scaled = loss / self.grad_accum

            # ── backward ──────────────────────────────────────────────────────
            # no GradScaler → plain backward
            loss_scaled.backward()

            # ── optimizer step every grad_accum batches ───────────────────────
            is_last_batch        = (step + 1) == len(self.train_dl)
            is_accum_step        = (step + 1) % self.grad_accum == 0

            if is_accum_step or is_last_batch:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.get('max_grad_norm', 1.0))

                self.opt.step()
                self.sched.step()
                self.opt.zero_grad()

            tot_loss  += loss.item()
            n_batches += 1

        return tot_loss / max(n_batches, 1)

    # ── validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        tot_loss = 0.0
        n_batches = 0
        all_preds, all_labels = [], []
        all_pred_idx, all_label_idx = [], []

        for batch in self.val_dl:
            ids    = batch['input_ids']     .to(self.device)
            mask   = batch['attention_mask'].to(self.device)
            labels = batch['label']         .to(self.device)

            with self.autocast_ctx:
                logits   = self.model(ids, mask)
                loss, *_ = self.loss_fn(logits, labels)

            pred_idx = logits.argmax(dim=1)
            all_pred_idx .extend(pred_idx.cpu().numpy())
            all_label_idx.extend(labels  .cpu().numpy())

            tot_loss  += loss.item()
            n_batches += 1

            all_preds  += ranked_preds(logits)
            all_labels += [ANSWER_LABELS[l.item()] for l in labels]

        m3        = map_at_3(all_preds, all_labels)
        acc       = accuracy_score(all_label_idx, all_pred_idx)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_label_idx, all_pred_idx,
            average='macro', zero_division=0)

        return tot_loss / max(n_batches, 1), m3, acc, prec, rec, f1

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self) -> dict:
        logger.info("DeBERTa fine-tuning start")

        for ep in range(1, self.cfg.get('epochs', 10) + 1):

            tr_loss = self._train_epoch()
            vl_loss, m3, acc, prec, rec, f1 = self._validate()

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
                # clone to CPU to free GPU memory
                self.best_state = {
                    k: v.clone().cpu()
                    for k, v in self.model.state_dict().items()
                }
                flag = '  ★'

            logger.info(
                f"Ep {ep:3d} | tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                f"MAP@3={m3:.4f}  Acc={acc:.4f}  lr={lr:.2e}{flag}")

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
                })

            if self._early_stop(m3):
                logger.info(f"Early stop at epoch {ep}")
                break

        # restore best weights (move back to device)
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