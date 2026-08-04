# src/DeBERTa/model.py
"""
DeBERTa-v3 MCQ model — fixed dtype handling.

Root cause of original bugs
────────────────────────────
1. bf16 autocast makes encoder output (cls) bf16
2. Head Linear layers stay fp32 (PyTorch default)
3. cls @ head.weight → dtype mismatch → RuntimeError at inference
4. During training the autocast context handled it silently but
   produced nan loss when loss tensor was read via .item()

Fix: explicitly cast cls → fp32 before the head.
     Head always runs in fp32 regardless of autocast context.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger("DeBERTa.Model")


class MCQDeBERTa(nn.Module):
    """
    Fine-tunes DeBERTa-v3 for 5-way MCQ.

    Parameters
    ──────────
    model_name         : HuggingFace model id
    classifier_dropout : dropout before the scoring head
    freeze_layers      : freeze first N transformer layers (0 = train all)
    """

    def __init__(
        self,
        model_name         : str   = "microsoft/deberta-v3-base",
        classifier_dropout : float = 0.1,
        freeze_layers      : int   = 0,
    ):
        super().__init__()

        from transformers import AutoModel, AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name)
        # do NOT set dropout in config — set it in head instead
        # (config dropout sometimes ignored in DeBERTa-v3)

        self.encoder = AutoModel.from_pretrained(
            model_name, config=hf_config)

        hidden = hf_config.hidden_size      # 768 for -base, 1024 for -large

        # ── classifier head — explicitly fp32 ────────────────────────────────
        # Head MUST stay fp32. We cast encoder output to fp32 before feeding
        # into head. This avoids the bf16/fp16 vs fp32 dtype mismatch.
        self.head = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )

        # force head to fp32 always
        self.head = self.head.float()

        self._init_head()

        if freeze_layers > 0:
            self._freeze(freeze_layers)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MCQDeBERTa | total={n_total:,}  trainable={n_train:,}")

    def _init_head(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _freeze(self, n_layers: int):
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        layers = getattr(self.encoder, 'encoder', None)
        if layers is not None:
            layer_list = getattr(layers, 'layer', None)
            if layer_list is not None:
                for layer in layer_list[:n_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False

        frozen = sum(
            p.numel() for p in self.parameters() if not p.requires_grad)
        logger.info(f"Frozen params: {frozen:,}  "
                    f"(first {n_layers} transformer layers + embeddings)")

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ──────────
        input_ids      : [B, N_options, L]
        attention_mask : [B, N_options, L]

        Returns
        ───────
        logits         : [B, N_options]  — always fp32
        """
        B, N, L = input_ids.shape

        ids  = input_ids     .view(B * N, L)
        mask = attention_mask.view(B * N, L)

        out = self.encoder(input_ids=ids, attention_mask=mask)

        # [B*N, hidden] — may be bf16/fp16 under autocast
        cls = out.last_hidden_state[:, 0, :]

        # ── CRITICAL FIX: cast to fp32 before head ────────────────────────────
        # Head weights are always fp32. If cls is bf16/fp16 (from autocast),
        # the matmul would crash or silently NaN. Casting here is safe and
        # cheap — only the [B*N, hidden] tensor is cast, not the whole model.
        cls = cls.float()

        logits = self.head(cls).view(B, N)   # [B, N]  fp32
        return logits