# src/RoBERTa/model.py
"""
RoBERTa-base MCQ model — memory-optimised for T4 x2.

Architecture
────────────
  Shared RoBERTa encoder  (one forward per option, weights shared)
  → Mean pooling           (memory-efficient; weighted-layer pool removed)
  → Multi-sample dropout   (n=3, reduced from 5)
  → Cross-option interaction transformer block
  → Scalar scorer per option → logits [B, 5]

Memory optimisations vs previous version
──────────────────────────────────────────
  ✗ WeightedLayerPooling   removed  (stored 12× hidden states = OOM)
  ✗ R-Drop                 removed  (2× backward graph = OOM)
  ✓ output_hidden_states   False    (saves 12× [B*5, L, H] tensors)
  ✓ gradient_checkpointing True     (recompute instead of store activations)
  ✓ n_dropouts             3        (was 5)
  ✓ All params float32              (no FP16 gradient errors)
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


class AttentionPooling(nn.Module):
    """Trainable context-vector attention over token representations."""
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1, bias=False),
        )

    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden).squeeze(-1)          # [B*5, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        w      = F.softmax(scores, dim=-1).unsqueeze(1) # [B*5, 1, L]
        return torch.bmm(w, hidden).squeeze(1)           # [B*5, H]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    Single-layer transformer block over the 5 option representations.

    Why this helps
    ──────────────
    MCQ is comparative — the correct answer must be distinguished from
    distractors.  Pure independent scoring misses inter-option relationships.
    Memory cost is negligible: sequence length = 5.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = hidden,
            num_heads   = 1,        # single head — 5-token sequence
            dropout     = dropout,
            batch_first = True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),   # no expansion — saves memory
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
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

    Benefits
    ────────
    - Implicit ensemble within one forward → better generalisation
    - No extra parameters vs a standard head
    - n=3 keeps memory overhead modest

    We compute all n passes in a single batched operation to avoid
    Python-loop overhead on GPU.
    """

    def __init__(self, hidden: int, n_dropouts: int = 3,
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
        """
        x : [B, 5, H] → [B, 5]
        Averages n_dropouts independent dropout realizations.
        """
        # stack dropout outputs → [n, B, 5, 1] → mean → [B, 5]
        out = torch.stack(
            [self.fc(drop(x)) for drop in self.dropouts], dim=0
        ).mean(dim=0).squeeze(-1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MCQRoBERTa(nn.Module):
    """
    RoBERTa-base MCQ scorer, memory-optimised for T4 x2.

    Forward
    ───────
    input_ids      : [B, 5, L]
    attention_mask : [B, 5, L]
    token_type_ids : [B, 5, L]  ← ignored (RoBERTa has no segment IDs)
    → logits       : [B, 5]

    Memory budget (per GPU, batch_size=4, max_len=96, FP32)
    ──────────────────────────────────────────────────────
    RoBERTa-base weights            ~440 MB
    Activations (grad ckpt)         ~1.5 GB  (recomputed, not stored)
    Optimizer states (AdamW)        ~880 MB
    Input tensors [4,5,96]          ~negligible
    Total estimated                 ~3.0 GB  ← fits in 15 GB T4
    """

    def __init__(
        self,
        model_name    : str   = "roberta-base",
        pooling       : str   = "mean",
        hidden_dropout: float = 0.1,
        n_dropouts    : int   = 3,
        use_grad_ckpt : bool  = True,
    ):
        super().__init__()

        # ── 1. RoBERTa backbone ───────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            output_hidden_states         = False,   # CRITICAL: saves 12× hidden states
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config      = cfg,
            torch_dtype = torch.float32,            # always FP32
        )

        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size          # 768 for roberta-base

        # ── 2. Pooling ────────────────────────────────────────────────────
        self.pooling_mode = pooling
        if pooling == "mean":
            self.pool = MeanPooling()
        elif pooling == "cls":
            self.pool = CLSPooling()
        elif pooling == "attention":
            self.pool = AttentionPooling(H)
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}. "
                             f"Choose from: mean, cls, attention")

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
        if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layer'):
            return list(enc.encoder.layer)
        return []

    def freeze_backbone_layers(self, n: int):
        """Freeze embeddings + bottom-n transformer layers."""
        layers = self._transformer_layers()
        if n >= len(layers):
            logger.warning(
                f"freeze_layers={n} >= total={len(layers)}. "
                f"Capping at {len(layers) - 1}."
            )
            n = max(0, len(layers) - 1)

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        frozen = sum(not p.requires_grad for p in self.encoder.parameters())
        total  = sum(1 for _ in self.encoder.parameters())
        logger.info(
            f"Frozen {frozen}/{total} backbone params "
            f"(bottom-{n} layers + embeddings)"
        )

    def unfreeze_top_layer(self) -> bool:
        """Unfreeze the topmost still-frozen transformer layer. Returns True if unfroze."""
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

        # flatten: treat each option as an independent sequence
        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)

        # RoBERTa does not use token_type_ids — omit entirely
        out = self.encoder(
            input_ids      = iids,
            attention_mask = mask,
        )                                           # last_hidden_state: [B*5, L, H]

        pooled = self.pool(out.last_hidden_state, mask)  # [B*5, H]

        # cross-option interaction
        pooled = pooled.view(B, N, -1)                   # [B, 5, H]
        pooled = self.option_interaction(pooled)          # [B, 5, H]

        # multi-sample dropout scoring
        logits = self.head(pooled)                        # [B, 5]
        return logits