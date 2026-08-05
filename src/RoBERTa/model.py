# src/RoBERTa/model.py
"""
RoBERTa-base MCQ model.

Changes from reviewed version
──────────────────────────────
  forward()
    - token_type_ids parameter renamed to token_type_ids=None with
      docstring noting it is accepted for API compatibility but ignored.

  freeze_backbone_layers / unfreeze_top_layer
    - Logic unchanged (was verified correct in review).

  _init_weights
    - LayerNorm left at PyTorch default (weight=1, bias=0) — correct.
    - nn.Linear xavier init unchanged.

  OptionInteraction
    - num_heads kept at 1: sequence length is 5 (options), so
      multi-head decomposition adds no benefit.
    - FFN 2× expansion kept (memory safe at these batch sizes).

  MultiSampleDropoutHead
    - n_dropouts=4 kept.

  No other architectural changes — MAP@3 = 0.80+ must be preserved.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

logger = logging.getLogger("RoBERTa.Model")


# ─────────────────────────────────────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────────────────────────────────────

class MeanPooling(nn.Module):
    """Attention-mask-aware mean over the token dimension."""

    def forward(
        self,
        hidden : torch.Tensor,   # [B, L, H]
        mask   : torch.Tensor,   # [B, L]
    ) -> torch.Tensor:           # [B, H]
        m      = mask.unsqueeze(-1).float()
        summed = (hidden * m).sum(dim=1)
        count  = m.sum(dim=1).clamp(min=1e-9)
        return summed / count


class CLSPooling(nn.Module):
    """Return the CLS token representation."""

    def forward(
        self,
        hidden : torch.Tensor,   # [B, L, H]
        mask   : torch.Tensor,   # [B, L]  (unused)
    ) -> torch.Tensor:           # [B, H]
        return hidden[:, 0, :]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    Single-layer transformer block over the 5 option representations.

    Sequence length = 5 (one vector per option) → negligible memory cost.
    Single attention head is sufficient: no benefit from multi-head at
    this sequence length.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = hidden,
            num_heads   = 1,
            dropout     = dropout,
            batch_first = True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 5, H] → [B, 5, H]"""
        attn_out, _ = self.attn(x, x, x)
        x           = self.norm1(x + self.drop(attn_out))
        x           = self.norm2(x + self.drop(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sample dropout head
# ─────────────────────────────────────────────────────────────────────────────

class MultiSampleDropoutHead(nn.Module):
    """
    Run the classification head *n_dropouts* times with different dropout
    masks and average the results.  Provides an implicit ensemble effect
    within a single forward pass.
    """

    def __init__(
        self,
        hidden     : int   = 768,
        n_dropouts : int   = 4,
        dropout_p  : float = 0.1,
    ):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for _ in range(n_dropouts)]
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 5, H] → [B, 5]"""
        return (
            torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0)
            .mean(dim=0)
            .squeeze(-1)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MCQRoBERTa(nn.Module):
    """
    RoBERTa-base MCQ scorer.

    Forward signature
    ─────────────────
    input_ids      : [B, 5, L]
    attention_mask : [B, 5, L]
    token_type_ids : [B, 5, L] | None
        Accepted for API compatibility with the DataParallel/collate
        pipeline; **silently ignored** — RoBERTa has no segment embeddings.

    Returns
    ───────
    logits : [B, 5]  — raw unnormalised scores (not softmaxed)
    """

    def __init__(
        self,
        model_name     : str   = "roberta-base",
        pooling        : str   = "mean",
        hidden_dropout : float = 0.1,
        n_dropouts     : int   = 4,
        use_grad_ckpt  : bool  = True,
    ):
        super().__init__()

        # ── 1. RoBERTa backbone ───────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            output_hidden_states         = False,
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config      = cfg,
            torch_dtype = torch.float32,
        )
        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size   # 768

        # ── 2. Pooling ────────────────────────────────────────────────────
        self.pooling_mode = pooling
        if pooling == "mean":
            self.pool = MeanPooling()
        elif pooling == "cls":
            self.pool = CLSPooling()
        else:
            raise ValueError(
                f"Unknown pooling={pooling!r}. Choose: mean | cls"
            )

        # ── 3. Cross-option interaction ───────────────────────────────────
        self.option_interaction = OptionInteraction(H, dropout=hidden_dropout)

        # ── 4. Multi-sample dropout head ──────────────────────────────────
        self.head = MultiSampleDropoutHead(
            hidden     = H,
            n_dropouts = n_dropouts,
            dropout_p  = hidden_dropout,
        )

        self._init_weights()

    # ── weight init ──────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform init for all Linear layers in head + interaction."""
        modules = (
            list(self.head.modules()) +
            list(self.option_interaction.modules())
        )
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── layer introspection ──────────────────────────────────────────────────

    def _transformer_layers(self) -> List[nn.Module]:
        """
        Return the list of transformer encoder layers in depth order.
        Robust to models with or without a pooler head.
        """
        enc = self.encoder
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
            return list(enc.encoder.layer)
        return []

    # ── freeze / unfreeze ────────────────────────────────────────────────────

    def freeze_backbone_layers(self, n: int) -> None:
        """
        Freeze the embedding layer and the bottom *n* transformer layers.

        After this call the optimizer should be built so it only includes
        parameters with requires_grad=True.
        """
        layers = self._transformer_layers()
        total  = len(layers)   # 12 for roberta-base

        if n >= total:
            logger.warning(
                "freeze_layers=%d >= total=%d. Capping at %d.",
                n, total, max(0, total - 1),
            )
            n = max(0, total - 1)

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        frozen  = sum(not p.requires_grad for p in self.encoder.parameters())
        total_p = sum(1 for _ in self.encoder.parameters())
        logger.info(
            "Frozen %d/%d encoder params "
            "(embeddings + bottom-%d of %d layers). "
            "Top %d layers trainable.",
            frozen, total_p, n, total, total - n,
        )

    def unfreeze_top_layer(self) -> bool:
        """
        Unfreeze the topmost still-frozen transformer layer.

        Iterates from the deepest layer upward; returns True if a layer
        was unfrozen, False if all layers are already trainable.

        Called via _get_raw_model(model).unfreeze_top_layer() to be
        DataParallel-safe.
        """
        for layer in reversed(self._transformer_layers()):
            if any(not p.requires_grad for p in layer.parameters()):
                for p in layer.parameters():
                    p.requires_grad = True
                return True
        return False

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids      : torch.Tensor,              # [B, 5, L]
        attention_mask : torch.Tensor,              # [B, 5, L]
        token_type_ids : Optional[torch.Tensor] = None,  # ignored
    ) -> torch.Tensor:                              # [B, 5]
        B, N, L = input_ids.shape

        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)

        out    = self.encoder(input_ids=iids, attention_mask=mask)
        pooled = self.pool(out.last_hidden_state, mask)   # [B*5, H]
        pooled = pooled.view(B, N, -1)                    # [B, 5, H]
        pooled = self.option_interaction(pooled)           # [B, 5, H]
        logits = self.head(pooled)                         # [B, 5]
        return logits