# src/RoBERTa/model.py
"""
RoBERTa-base MCQ model.

Architecture
────────────
  Shared RoBERTa encoder  (one forward per option, weights shared)
  → Mean pooling
  → Multi-sample dropout head  (n=4)
  → Cross-option interaction transformer block
  → Scalar scorer per option → logits [B, 5]

Fixes vs previous version
──────────────────────────
  - output_hidden_states=False  (no OOM from storing all layer states)
  - n_dropouts=4 (was 3 in OOM-fix version, now 4 for better ensemble)
  - OptionInteraction FFN expanded back to 2× (was 1× in OOM-fix)
    because we have enough memory now with correct batch/len settings
  - _transformer_layers() made robust (handles pooler layer presence)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

logger = logging.getLogger("RoBERTa.Model")


# ─────────────────────────────────────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────────────────────────────────────

class MeanPooling(nn.Module):
    """Attention-mask-aware mean over token dimension."""
    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        m      = mask.unsqueeze(-1).float()
        summed = (hidden * m).sum(dim=1)
        count  = m.sum(dim=1).clamp(min=1e-9)
        return summed / count


class CLSPooling(nn.Module):
    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0, :]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    Single-layer transformer block over the 5 option representations.

    MCQ is comparative: the correct answer must be distinguished from
    distractors. Pure independent scoring misses inter-option relationships.
    Memory cost negligible (sequence length = 5).
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
            nn.Linear(hidden, hidden * 2),   # 2× expansion restored
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
    Run the classifier head n times with different dropout masks, average.

    Provides implicit ensemble effect within one forward pass.
    n=4 balances generalisation vs compute.
    """

    def __init__(self, hidden: int, n_dropouts: int = 4,
                 dropout_p: float = 0.1):
        super().__init__()
        self.n_dropouts = n_dropouts
        self.dropouts   = nn.ModuleList([
            nn.Dropout(dropout_p) for _ in range(n_dropouts)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 5, H] → [B, 5]"""
        return torch.stack(
            [self.fc(drop(x)) for drop in self.dropouts], dim=0
        ).mean(dim=0).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MCQRoBERTa(nn.Module):
    """
    RoBERTa-base MCQ scorer.

    Forward
    ───────
    input_ids      : [B, 5, L]
    attention_mask : [B, 5, L]
    token_type_ids : [B, 5, L]  ← ignored (RoBERTa has no segment IDs)
    → logits       : [B, 5]
    """

    def __init__(
        self,
        model_name    : str   = "roberta-base",
        pooling       : str   = "mean",
        hidden_dropout: float = 0.1,
        n_dropouts    : int   = 4,
        use_grad_ckpt : bool  = True,
    ):
        super().__init__()

        # ── 1. RoBERTa backbone ───────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            output_hidden_states         = False,  # no OOM from storing all layers
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config      = cfg,
            torch_dtype = torch.float32,           # always FP32
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
            raise ValueError(f"Unknown pooling: {pooling!r}. "
                             f"Choose: mean | cls")

        # ── 3. Cross-option interaction ───────────────────────────────────
        self.option_interaction = OptionInteraction(H, dropout=hidden_dropout)

        # ── 4. Multi-sample dropout head ──────────────────────────────────
        self.head = MultiSampleDropoutHead(
            hidden     = H,
            n_dropouts = n_dropouts,
            dropout_p  = hidden_dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in [
            *list(self.head.modules()),
            *list(self.option_interaction.modules()),
        ]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── layer freeze / unfreeze ───────────────────────────────────────────────

    def _transformer_layers(self) -> list:
        """
        Return the list of transformer encoder layers.
        Robust to the presence of a pooler layer.
        """
        enc = self.encoder
        if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
            return list(enc.encoder.layer)
        return []

    def freeze_backbone_layers(self, n: int):
        """Freeze embeddings + bottom-n transformer layers."""
        layers = self._transformer_layers()
        total  = len(layers)     # 12 for roberta-base

        if n >= total:
            logger.warning(
                f"freeze_layers={n} >= total={total}. "
                f"Capping at {total - 1}."
            )
            n = max(0, total - 1)

        # freeze embeddings
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        # freeze bottom n layers
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        frozen  = sum(not p.requires_grad for p in self.encoder.parameters())
        total_p = sum(1 for _ in self.encoder.parameters())
        logger.info(
            f"Frozen {frozen}/{total_p} encoder params "
            f"(embeddings + bottom-{n} of {total} layers). "
            f"Top {total - n} layers are trainable."
        )

    def unfreeze_top_layer(self) -> bool:
        """
        Unfreeze the topmost still-frozen transformer layer.
        Returns True if a layer was unfrozen, False if all are already trainable.
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
        input_ids      : torch.Tensor,   # [B, 5, L]
        attention_mask : torch.Tensor,   # [B, 5, L]
        token_type_ids : torch.Tensor,   # [B, 5, L]  ignored
    ) -> torch.Tensor:                   # → [B, 5]

        B, N, L = input_ids.shape

        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)

        out = self.encoder(
            input_ids      = iids,
            attention_mask = mask,
        )

        pooled = self.pool(out.last_hidden_state, mask)  # [B*5, H]
        pooled = pooled.view(B, N, -1)                   # [B, 5, H]
        pooled = self.option_interaction(pooled)          # [B, 5, H]
        logits = self.head(pooled)                        # [B, 5]
        return logits