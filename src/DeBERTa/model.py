# src/DeBERTa/model.py
"""
DeBERTa-v3-small MCQ model.

Architecture
────────────
  Shared DeBERTa encoder  (one forward per option, weights shared)
  → Pooling  (mean / cls / attention-weighted)
  → Option-pair interaction layer   ← NEW: attends across all 5 options
  → Scalar scorer per option
  → logits [B, 5]

Option Interaction Layer
────────────────────────
After getting one rep per option, we apply a small cross-option
transformer so the model learns "option A is better than B given C".
This is the key addition over naive independent scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# ─────────────────────────────────────────────────────────────────────────────
# Pooling heads
# ─────────────────────────────────────────────────────────────────────────────

class MeanPooling(nn.Module):
    """Mean of non-padding token representations."""
    def forward(self, last_hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        m     = mask.unsqueeze(-1).float()          # [B, L, 1]
        summed = (last_hidden * m).sum(dim=1)       # [B, H]
        count  = m.sum(dim=1).clamp(min=1e-9)       # [B, 1]
        return summed / count                        # [B, H]


class CLSPooling(nn.Module):
    def forward(self, last_hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return last_hidden[:, 0, :]                 # [B, H]


class AttentionPooling(nn.Module):
    """Trainable context-vector attention over token representations."""
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1, bias=False),
        )

    def forward(self, last_hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(last_hidden).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        w      = F.softmax(scores, dim=-1).unsqueeze(1)  # [B, 1, L]
        return torch.bmm(w, last_hidden).squeeze(1)      # [B, H]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction — small single-head transformer block
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    Takes [B, 5, H] option representations and lets options
    attend to each other before scoring.
    Very lightweight: 1 attention head + FFN.

    Why this helps
    ──────────────
    MCQ is inherently comparative — the correct option must be
    distinguished from distractors. Pure independent scoring misses
    inter-option relationships.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = hidden,
            num_heads   = 1,          # single head — options sequence is short (5)
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
        """
        x : [B, 5, H]
        returns: [B, 5, H]
        """
        # self-attention across the 5 options
        attn_out, _ = self.attn(x, x, x)
        x           = self.norm1(x + self.drop(attn_out))
        # position-wise FFN
        x           = self.norm2(x + self.drop(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MCQDeBERTa(nn.Module):
    """
    DeBERTa-v3-small MCQ scorer with cross-option interaction.

    Forward
    ───────
    input_ids      : [B, 5, L]
    attention_mask : [B, 5, L]
    token_type_ids : [B, 5, L]
    → logits       : [B, 5]
    """

    def __init__(
        self,
        model_name    : str   = "microsoft/deberta-v3-small",
        pooling       : str   = "mean",
        hidden_dropout: float = 0.1,
        use_grad_ckpt : bool  = True,
    ):
        super().__init__()

        # ── 1. DeBERTa backbone ───────────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            output_hidden_states         = False,
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=torch.float32,
        )

        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size   # 768 for deberta-v3-small

        # ── 2. Pooling ─────────────────────────────────────────────────────────
        self.pooling_mode = pooling
        if pooling == "mean":
            self.pool = MeanPooling()
        elif pooling == "cls":
            self.pool = CLSPooling()
        elif pooling == "attention":
            self.pool = AttentionPooling(H)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # ── 3. Cross-option interaction ────────────────────────────────────────
        self.option_interaction = OptionInteraction(H, dropout=hidden_dropout)

        # ── 4. Classifier head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.LayerNorm(H // 2),
            nn.Linear(H // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [*list(self.head.modules()),
                  *list(self.option_interaction.modules())]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── layer freeze / unfreeze helpers ───────────────────────────────────────

    def _transformer_layers(self):
        enc = self.encoder
        if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
            return list(enc.encoder.layer)
        return []

    def freeze_backbone_layers(self, n: int):
        """Freeze embeddings + bottom-n transformer layers."""
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        for layer in self._transformer_layers()[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        frozen  = sum(not p.requires_grad for p in self.encoder.parameters())
        total   = sum(1 for _ in self.encoder.parameters())
        logger_msg = (
            f"Frozen {frozen}/{total} backbone params "
            f"(bottom-{n} layers + embeddings)"
        )
        import logging
        logging.getLogger("DeBERTa.Model").info(logger_msg)

    def unfreeze_top_layer(self) -> bool:
        """Unfreeze the topmost still-frozen transformer layer."""
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
        token_type_ids : torch.Tensor,   # [B, 5, L]
    ) -> torch.Tensor:                   # → [B, 5]

        B, N, L = input_ids.shape

        # ── encode all options jointly (flatten → encode → reshape) ───────────
        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)
        tids = token_type_ids.view(B * N, L)

        out         = self.encoder(
            input_ids      = iids,
            attention_mask = mask,
            token_type_ids = tids,
        )
        last_hidden = out.last_hidden_state              # [B*5, L, H]
        pooled      = self.pool(last_hidden, mask)       # [B*5, H]

        # ── cross-option interaction ──────────────────────────────────────────
        pooled = pooled.view(B, N, -1)                   # [B, 5, H]
        pooled = self.option_interaction(pooled)         # [B, 5, H]

        # ── score each option ─────────────────────────────────────────────────
        logits = self.head(pooled).squeeze(-1)           # [B, 5]
        return logits