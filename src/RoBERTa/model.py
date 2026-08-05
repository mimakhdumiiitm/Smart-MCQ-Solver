# src/RoBERTa/model.py
"""
RoBERTa-base MCQ model.

Architecture
────────────
  Shared RoBERTa encoder   (one forward per option, weights shared)
  → Weighted-layer pooling (learnable combination of last-K hidden states)
  → Multi-sample dropout   (average over K dropout masks → reduces variance)
  → Cross-option interaction transformer block
  → Scalar scorer per option  →  logits [B, 5]

Design choices to beat MAP@3 0.788
────────────────────────────────────
1. Weighted layer pooling  — last 4 layers capture different
   abstraction levels; letting the model learn the mix beats
   always using the last layer.

2. Multi-sample dropout    — averages K forward passes through
   the head with different dropout masks.  Equivalent to an
   implicit ensemble; significantly reduces val variance.

3. Cross-option interaction — same motivation as DeBERTa version.
   MCQ is comparative; options must attend to each other.

4. Careful dtype handling  — all parameters stay float32.
   AMP is applied only to the encoder forward pass.
   GradScaler works in float32 mode (no FP16 parameter casting).
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
    """Masked mean of token representations."""
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m      = mask.unsqueeze(-1).float()
        summed = (hidden * m).sum(1)
        count  = m.sum(1).clamp(min=1e-9)
        return summed / count


class CLSPooling(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0, :]


class AttentionPooling(nn.Module):
    """Trainable attention-weighted pooling."""
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1, bias=False),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w      = F.softmax(scores, dim=-1).unsqueeze(1)
        return torch.bmm(w, hidden).squeeze(1)


class WeightedLayerPooling(nn.Module):
    """
    Learnable weighted combination of the last `n_layers` hidden states.

    This consistently outperforms single-layer pooling on downstream
    tasks because different layers encode syntax (lower) vs. semantics
    (upper) differently.

    hidden_states : tuple of [B, L, H]  (all transformer layer outputs)
    """

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers
        # raw weights → softmax normalised inside forward
        self.layer_weights = nn.Parameter(torch.ones(n_layers))

    def forward(
        self,
        hidden_states: tuple,   # (n_total_layers+1,) each [B, L, H]
        mask: torch.Tensor,     # [B, L]
    ) -> torch.Tensor:          # [B, H]
        # take last n_layers (skip embedding layer at index 0)
        layers = hidden_states[-self.n_layers:]   # list of [B, L, H]
        stacked = torch.stack(layers, dim=0)      # [n_layers, B, L, H]

        w = F.softmax(self.layer_weights, dim=0)  # [n_layers]
        # weighted sum across layer dimension
        weighted = (stacked * w[:, None, None, None]).sum(0)  # [B, L, H]

        # masked mean over token dimension
        m      = mask.unsqueeze(-1).float()
        summed = (weighted * m).sum(1)
        count  = m.sum(1).clamp(min=1e-9)
        return summed / count                     # [B, H]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-option interaction
# ─────────────────────────────────────────────────────────────────────────────

class OptionInteraction(nn.Module):
    """
    [B, 5, H] → [B, 5, H]

    Single-head self-attention over the 5 option representations
    followed by a position-wise FFN.  Lets options compete/contrast.
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
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sample dropout head
# ─────────────────────────────────────────────────────────────────────────────

class MultiSampleDropoutHead(nn.Module):
    """
    Averages K dropout samples to reduce output variance.

    Reference: Inoue (2019) "Multi-Sample Dropout for Accelerated
    Training and Better Generalization"

    Each sample uses an independent Dropout mask.  The averaged
    logit has lower variance than a single-mask prediction without
    requiring any extra parameters.
    """

    def __init__(
        self,
        hidden     : int,
        n_samples  : int   = 5,
        p_low      : float = 0.10,
        p_high     : float = 0.50,
    ):
        super().__init__()
        self.n_samples = n_samples
        # evenly-spaced dropout rates between p_low and p_high
        ps = [p_low + (p_high - p_low) * i / max(n_samples - 1, 1)
              for i in range(n_samples)]
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in ps])
        self.fc = nn.Linear(hidden, hidden // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden // 2)
        self.out = nn.Linear(hidden // 2, 1)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.fc.weight);  nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H]  →  [B, 1]"""
        outs = []
        for drop in self.dropouts:
            h = self.act(self.norm(self.fc(drop(x))))
            outs.append(self.out(h))
        return torch.stack(outs, dim=0).mean(dim=0)  # [B, 1]


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
    token_type_ids : [B, 5, L]   ← ignored (zeros), kept for API compat
    → logits       : [B, 5]

    Dtype policy
    ────────────
    All parameters are float32.  AMP autocast is applied externally
    only to the encoder call (see Trainer).  GradScaler operates on
    float32 gradients — no "Attempting to unscale FP16 gradients" error.
    """

    def __init__(
        self,
        model_name     : str   = "roberta-base",
        pooling        : str   = "weighted_layer",
        hidden_dropout : float = 0.1,
        use_grad_ckpt  : bool  = True,
        n_weighted_layers : int  = 4,
        multi_sample_dropout : bool  = True,
        n_dropout_samples    : int   = 5,
        dropout_low          : float = 0.10,
        dropout_high         : float = 0.50,
    ):
        super().__init__()

        # ── 1. RoBERTa backbone ───────────────────────────────────────────────
        cfg = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob          = hidden_dropout,
            attention_probs_dropout_prob = hidden_dropout,
            # MUST return all hidden states for weighted-layer pooling
            output_hidden_states         = (pooling == "weighted_layer"),
        )
        # Load in float32 explicitly
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config     = cfg,
            torch_dtype = torch.float32,
        )

        if use_grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size  # 768 for roberta-base

        # ── 2. Pooling ─────────────────────────────────────────────────────────
        self.pooling_mode = pooling
        self.n_weighted_layers = n_weighted_layers

        if pooling == "weighted_layer":
            self.pool = WeightedLayerPooling(n_weighted_layers)
        elif pooling == "mean":
            self.pool = MeanPooling()
        elif pooling == "cls":
            self.pool = CLSPooling()
        elif pooling == "attention":
            self.pool = AttentionPooling(H)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # ── 3. Cross-option interaction ────────────────────────────────────────
        self.option_interaction = OptionInteraction(H, dropout=hidden_dropout)

        # ── 4. Scoring head ────────────────────────────────────────────────────
        self.use_msd = multi_sample_dropout
        if multi_sample_dropout:
            self.head = MultiSampleDropoutHead(
                hidden    = H,
                n_samples = n_dropout_samples,
                p_low     = dropout_low,
                p_high    = dropout_high,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(H, H // 2),
                nn.GELU(),
                nn.Dropout(hidden_dropout),
                nn.LayerNorm(H // 2),
                nn.Linear(H // 2, 1),
            )

        self._init_non_pretrained()
        logger.info(
            f"MCQRoBERTa | pooling={pooling} | "
            f"msd={multi_sample_dropout} | H={H}"
        )

    def _init_non_pretrained(self):
        """Xavier-init all non-pretrained linear layers."""
        modules = (
            list(self.option_interaction.modules()) +
            list(self.head.modules()) +
            (list(self.pool.modules())
             if hasattr(self.pool, 'modules') else [])
        )
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── layer freeze / unfreeze helpers ───────────────────────────────────────

    def _transformer_layers(self):
        enc = self.encoder
        # RoBERTa: encoder.encoder.layer
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

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids      : torch.Tensor,  # [B, 5, L]
        attention_mask : torch.Tensor,  # [B, 5, L]
        token_type_ids : torch.Tensor,  # [B, 5, L]  ignored by RoBERTa
    ) -> torch.Tensor:                  # → [B, 5]

        B, N, L = input_ids.shape

        # flatten batch × options for joint encoding
        iids = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)
        # RoBERTa doesn't use token_type_ids; don't pass them

        out = self.encoder(
            input_ids      = iids,
            attention_mask = mask,
        )

        # ── pooling ───────────────────────────────────────────────────────────
        if self.pooling_mode == "weighted_layer":
            # out.hidden_states is a tuple: (embedding, layer_1, …, layer_12)
            pooled = self.pool(out.hidden_states, mask)   # [B*5, H]
        else:
            pooled = self.pool(out.last_hidden_state, mask)  # [B*5, H]

        # ── cross-option interaction ──────────────────────────────────────────
        pooled = pooled.view(B, N, -1)               # [B, 5, H]
        pooled = self.option_interaction(pooled)     # [B, 5, H]

        # ── score each option ─────────────────────────────────────────────────
        # reshape for head: [B*5, H]
        pooled_flat = pooled.view(B * N, -1)
        logits_flat = self.head(pooled_flat)         # [B*5, 1]
        logits      = logits_flat.view(B, N)         # [B, 5]

        return logits