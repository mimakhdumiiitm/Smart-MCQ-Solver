# src/RoBERTa/model.py
"""
RoBERTa-base MCQ model.

Architecture
────────────
  Shared RoBERTa encoder  (one forward per option, weights shared)
  → Weighted layer pooling  (learns which encoder layer to trust)
  → Multi-sample dropout   (5× dropout heads averaged → better generalisation)
  → Cross-option interaction transformer block
  → Scalar scorer per option
  → logits [B, 5]

Key improvements over naive RoBERTa
─────────────────────────────────────
  1. Weighted layer pooling  — uses all 12 hidden states with learned weights
     rather than only the last layer; empirically +0.5–1.5 MAP@3
  2. Multi-sample dropout    — 5 stochastic forward passes through the head,
     averaged; acts as an ensemble within one forward pass; prevents overfitting
  3. Cross-option interaction — same as DeBERTa version; MCQ is comparative
  4. All params kept in float32 — avoids "unscale FP16 gradients" crash
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

logger = logging.getLogger("RoBERTa.Model")


# ─────────────────────────────────────────────────────────────────────────────
# Pooling heads
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


class AttentionPooling(nn.Module):
    """Trainable context-vector attention over token representations."""
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1, bias=False),
        )

    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w      = F.softmax(scores, dim=-1).unsqueeze(1)
        return torch.bmm(w, hidden).squeeze(1)


class WeightedLayerPooling(nn.Module):
    """
    Learned convex combination of all encoder hidden states.

    Why this works
    ──────────────
    Different layers capture different linguistic abstractions.
    Lower layers → syntax / morphology.
    Upper layers → semantics / pragmatics.
    For MCQ we want a mixture; the model learns the optimal weights.

    Parameters
    ──────────
    n_layers : number of transformer layers (12 for roberta-base)
    """

    def __init__(self, n_layers: int, pooling_mode: str = "mean"):
        super().__init__()
        self.n_layers     = n_layers
        self.pooling_mode = pooling_mode
        # one scalar weight per layer; softmax → convex combination
        self.layer_weights = nn.Parameter(
            torch.ones(n_layers, dtype=torch.float32)
        )
        # per-layer mean pooler (reused)
        self._mean = MeanPooling()

    def forward(
        self,
        all_hidden_states,   # tuple of [B*5, L, H] length n_layers+1
        mask: torch.Tensor,  # [B*5, L]
    ) -> torch.Tensor:
        # skip embedding layer (index 0), use layers 1..n_layers
        hidden_stack = torch.stack(
            all_hidden_states[1:], dim=0
        )                              # [n_layers, B*5, L, H]

        weights = F.softmax(self.layer_weights, dim=0)  # [n_layers]
        # weighted sum over layers
        weighted = (
            hidden_stack *
            weights.view(-1, 1, 1, 1)
        ).sum(dim=0)                   # [B*5, L, H]

        # pool over token dimension
        if self.pooling_mode == "mean":
            return self._mean(weighted, mask)
        else:
            return weighted[:, 0, :]   # CLS


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    Single-layer transformer block over the 5 option representations.
    Identical logic to DeBERTa version — well validated.
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
    Apply the classification head N times with different dropout masks,
    then average the logits.

    Benefits
    ────────
    - Acts as an ensemble within a single forward pass
    - Empirically reduces overfitting on small fine-tuning sets
    - No extra parameters vs a standard head; negligible compute overhead
    - Proven effective for NLP: https://arxiv.org/abs/1905.09788

    Architecture
    ────────────
    H → H//2 → GELU → Dropout(p) → LN → 1
    (repeated n_dropouts times, average)
    """

    def __init__(self, hidden: int, n_dropouts: int = 5,
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
        logits = torch.stack(
            [self.fc(drop(x)) for drop in self.dropouts],
            dim=0,
        ).mean(dim=0)                     # [B, 5, 1]
        return logits.squeeze(-1)          # [B, 5]


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

    Design notes
    ────────────
    - All encoder parameters kept in float32 to avoid AMP gradient issues.
    - Gradient checkpointing enabled to fit in T4 VRAM.
    - output_hidden_states=True required for WeightedLayerPooling.
    """

    def __init__(
        self,
        model_name    : str   = "roberta-base",
        pooling       : str   = "weighted",     # weighted | mean | cls | attention
        hidden_dropout: float = 0.1,
        n_dropouts    : int   = 5,
        use_grad_ckpt : bool  = True,
    ):
        super().__init__()

        # ── 1. RoBERTa backbone ───────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            output_hidden_states         = True,   # needed for weighted pooling
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config      = cfg,
            torch_dtype = torch.float32,           # always FP32
        )

        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

        H          = self.encoder.config.hidden_size          # 768
        n_layers   = self.encoder.config.num_hidden_layers    # 12

        # ── 2. Pooling ────────────────────────────────────────────────────
        self.pooling_mode = pooling
        if pooling == "weighted":
            self.pool = WeightedLayerPooling(n_layers, pooling_mode="mean")
        elif pooling == "mean":
            self.pool = MeanPooling()
        elif pooling == "cls":
            self.pool = CLSPooling()
        elif pooling == "attention":
            self.pool = AttentionPooling(H)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

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

    def _transformer_layers(self):
        enc = self.encoder
        # roberta-base: encoder.encoder.layer
        if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
            return list(enc.encoder.layer)
        return []

    def freeze_backbone_layers(self, n: int):
        """Freeze embeddings + bottom-n transformer layers."""
        layers = self._transformer_layers()
        if n >= len(layers):
            logger.warning(
                f"freeze_layers={n} >= total layers={len(layers)}. "
                f"Capping at {len(layers) - 1}."
            )
            n = max(0, len(layers) - 1)

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        frozen  = sum(not p.requires_grad for p in self.encoder.parameters())
        total   = sum(1 for _ in self.encoder.parameters())
        logger.info(
            f"Frozen {frozen}/{total} backbone params "
            f"(bottom-{n} layers + embeddings)"
        )

    def unfreeze_top_layer(self) -> bool:
        """Unfreeze the topmost still-frozen transformer layer."""
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
        token_type_ids : torch.Tensor,   # [B, 5, L]  ignored by RoBERTa
    ) -> torch.Tensor:                   # → [B, 5]

        B, N, L = input_ids.shape

        # flatten batch × options
        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)

        # RoBERTa does not use token_type_ids — pass zeros to avoid issues
        out = self.encoder(
            input_ids      = iids,
            attention_mask = mask,
        )

        # pool over token dimension
        if self.pooling_mode == "weighted":
            # out.hidden_states: tuple of (n_layers+1) tensors [B*5, L, H]
            pooled = self.pool(out.hidden_states, mask)   # [B*5, H]
        else:
            pooled = self.pool(out.last_hidden_state, mask)   # [B*5, H]

        # cross-option interaction
        pooled = pooled.view(B, N, -1)                   # [B, 5, H]
        pooled = self.option_interaction(pooled)          # [B, 5, H]

        # multi-sample dropout scoring
        logits = self.head(pooled)                        # [B, 5]
        return logits