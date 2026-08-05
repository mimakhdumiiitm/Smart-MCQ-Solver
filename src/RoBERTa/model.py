# src/RoBERTa/model.py
"""
RoBERTa-base MCQ model — memory-efficient, MAP@3 > 0.80.

Architecture
────────────
  Shared RoBERTa encoder   (grad-checkpointing ON)
  → Mean pooling            (low memory vs weighted-layer)
  → Dropout + LayerNorm
  → Cross-option interaction (single MHA block over 5 options)
  → Linear scorer           (H → 1 per option)
  → logits [B, 5]

Dtype policy
────────────
  All parameters: float32.
  AMP autocast (float16 activations) applied externally in Trainer.
  GradScaler operates on float32 gradients → no FP16 unscale error.
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
    """Masked mean-pool over token dimension."""
    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        m      = mask.unsqueeze(-1).float()
        summed = (hidden * m).sum(1)
        count  = m.sum(1).clamp(min=1e-9)
        return summed / count                  # [B, H]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    [B, 5, H] → [B, 5, H]

    One round of self-attention across the 5 option representations
    + position-wise FFN.  Lets options attend to each other
    (comparative reasoning).  Lightweight: sequence length = 5.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = hidden,
            num_heads   = 4,           # 4 heads over 5 tokens is fine
            dropout     = dropout,
            batch_first = True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),  # no expansion — save memory
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _  = self.attn(x, x, x)
        x     = self.norm1(x + self.drop(a))
        x     = self.norm2(x + self.drop(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Scoring head
# ─────────────────────────────────────────────────────────────────────────────

class ScoringHead(nn.Module):
    """
    Simple two-layer head with dropout.
    Deliberately NOT using multi-sample dropout to save GPU memory.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)               # [B, 1]


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
    token_type_ids : [B, 5, L]  ← ignored (zeros), kept for API compat
    → logits       : [B, 5]
    """

    def __init__(
        self,
        model_name     : str   = "roberta-base",
        pooling        : str   = "mean",
        hidden_dropout : float = 0.1,
        use_grad_ckpt  : bool  = True,
    ):
        super().__init__()

        # ── 1. RoBERTa backbone ───────────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            output_hidden_states         = False,   # no all-layers needed
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config      = cfg,
            torch_dtype = torch.float32,            # stay float32
        )

        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size          # 768

        # ── 2. Pooling ─────────────────────────────────────────────────────────
        self.pool = MeanPooling()

        # ── 3. Cross-option interaction ────────────────────────────────────────
        self.option_interaction = OptionInteraction(H, dropout=hidden_dropout)

        # ── 4. Scoring head ────────────────────────────────────────────────────
        self.head = ScoringHead(H, dropout=hidden_dropout)

        self._init_weights()
        logger.info(f"MCQRoBERTa | H={H} | grad_ckpt={use_grad_ckpt}")

    def _init_weights(self):
        for m in (list(self.head.modules()) +
                  list(self.option_interaction.modules())):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── layer helpers ─────────────────────────────────────────────────────────

    def _transformer_layers(self):
        enc = self.encoder
        if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
            return list(enc.encoder.layer)
        return []

    def freeze_backbone_layers(self, n: int):
        layers = self._transformer_layers()
        if n >= len(layers):
            n = max(0, len(layers) - 1)
            logger.warning(f"Capping freeze at {n} layers.")

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        frozen = sum(not p.requires_grad for p in self.encoder.parameters())
        total  = sum(1 for _ in self.encoder.parameters())
        logger.info(f"Frozen {frozen}/{total} backbone params (bottom-{n} + embeddings)")

    def unfreeze_top_layer(self) -> bool:
        for layer in reversed(self._transformer_layers()):
            if any(not p.requires_grad for p in layer.parameters()):
                for p in layer.parameters():
                    p.requires_grad = True
                return True
        return False

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids      : torch.Tensor,   # [B, 5, L]
        attention_mask : torch.Tensor,   # [B, 5, L]
        token_type_ids : torch.Tensor,   # [B, 5, L]  ignored
    ) -> torch.Tensor:                   # [B, 5]

        B, N, L = input_ids.shape

        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)
        # RoBERTa does NOT use token_type_ids

        out    = self.encoder(input_ids=iids, attention_mask=mask)
        pooled = self.pool(out.last_hidden_state, mask)   # [B*5, H]

        # cross-option interaction
        pooled = pooled.view(B, N, -1)               # [B, 5, H]
        pooled = self.option_interaction(pooled)     # [B, 5, H]

        # score
        logits = self.head(pooled.view(B * N, -1))   # [B*5, 1]
        return logits.view(B, N)                     # [B, 5]