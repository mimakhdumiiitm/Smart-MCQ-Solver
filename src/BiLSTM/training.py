# training.py
"""
All training logic: loss + metrics + early stopping + trainer
"""

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Trainer")

ANSWER_LABELS = ['A', 'B', 'C', 'D', 'E']


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class MCQLoss(nn.Module):
    def __init__(self, smoothing=0.15, margin=0.5, ce_w=0.6, rank_w=0.4):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.margin = margin
        self.ce_w   = ce_w
        self.rank_w = rank_w

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets)
        pos       = logits.gather(1, targets.unsqueeze(1))
        margin    = F.relu(self.margin - (pos - logits))
        eye       = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.)
        rank_loss = (margin * (1 - eye)).sum(1).mean()
        total     = self.ce_w * ce_loss + self.rank_w * rank_loss
        return total, ce_loss.item(), rank_loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def ranked_preds(logits):
    top3 = torch.argsort(logits, dim=-1, descending=True)[:, :3]
    return [[ANSWER_LABELS[i.item()] for i in row] for row in top3]

def map_at_3(preds, labels):
    scores = []
    for pred, gold in zip(preds, labels):
        ap, hits = 0., 0
        for k, p in enumerate(pred[:3], 1):
            if p == gold:
                hits += 1; ap += hits / k
        scores.append(ap)
    return float(np.mean(scores))

def top1_acc(preds, labels):
    return float(np.mean([p[0] == g for p, g in zip(preds, labels)]))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer  (early stopping included)
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model, train_dl, val_dl,
                 opt, sched, loss_fn, device, cfg, wandb_run=None):
        self.model     = model.to(device)
        self.train_dl  = train_dl
        self.val_dl    = val_dl
        self.opt       = opt
        self.sched     = sched
        self.loss_fn   = loss_fn
        self.device    = device
        self.cfg       = cfg
        self.wandb_run = wandb_run

        self.history    = defaultdict(list)
        self.best_map3  = -np.inf
        self.best_state = None
        # early stopping state — kept inline (no separate class needed)
        self._es_counter = 0
        self._es_best    = -np.inf

    def _early_stop(self, score) -> bool:
        patience  = self.cfg.get('early_stop_patience', 8)
        min_delta = 1e-4
        if score > self._es_best + min_delta:
            self._es_best = score; self._es_counter = 0; return False
        self._es_counter += 1
        return self._es_counter >= patience

    def _train_epoch(self):
        self.model.train()
        tot = n = 0
        for batch in self.train_dl:
            opts   = batch['options'].to(self.device)
            lens   = batch['lengths'].to(self.device)
            labels = batch['label'].to(self.device)
            loss, *_ = self.loss_fn(self.model(opts, lens), labels)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get('max_grad_norm', 1.0))
            self.opt.step()
            tot += loss.item(); n += 1
        return tot / max(n, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        tot = n = 0
        all_preds, all_labels = [], []
        for batch in self.val_dl:
            opts   = batch['options'].to(self.device)
            lens   = batch['lengths'].to(self.device)
            labels = batch['label'].to(self.device)
            logits = self.model(opts, lens)
            loss, *_ = self.loss_fn(logits, labels)
            tot += loss.item(); n += 1
            all_preds  += ranked_preds(logits)
            all_labels += [ANSWER_LABELS[l.item()] for l in labels]
        m3  = map_at_3(all_preds, all_labels)
        acc = top1_acc(all_preds, all_labels)
        return tot / max(n, 1), m3, acc

    def train(self):
        logger.info("Training start")
        for ep in range(1, self.cfg.get('epochs', 40) + 1):
            tr_loss          = self._train_epoch()
            vl_loss, m3, acc = self._validate()

            if isinstance(self.sched,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched.step(m3)
            else:
                self.sched.step()

            lr = self.opt.param_groups[0]['lr']
            for k, v in [('tr_loss', tr_loss), ('vl_loss', vl_loss),
                         ('vl_map3', m3), ('vl_acc', acc), ('lr', lr)]:
                self.history[k].append(v)

            flag = ''
            if m3 > self.best_map3:
                self.best_map3  = m3
                self.best_state = {k: v.clone()
                                   for k, v in self.model.state_dict().items()}
                flag = '  ★'

            logger.info(f"Ep {ep:3d} | tr={tr_loss:.4f} vl={vl_loss:.4f} "
                        f"MAP@3={m3:.4f} Acc={acc:.4f} lr={lr:.1e}{flag}")

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch": ep, "train/loss": tr_loss,
                    "val/loss": vl_loss, "val/map3": m3,
                    "val/acc": acc, "lr": lr,
                }, step=ep)

            if self._early_stop(m3):
                logger.info(f"Early stop at epoch {ep}"); break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        logger.info(f"Best Val MAP@3 = {self.best_map3:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.summary.update({
                'best_val_map3': self.best_map3,
                'best_val_acc':  max(self.history['vl_acc']),
            })

        return dict(self.history)