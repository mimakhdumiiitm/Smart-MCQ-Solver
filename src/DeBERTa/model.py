# src/DeBERTa/model.py
"""
DeBERTa-v3 MCQ model.

Architecture
────────────
• DeBERTa-v3 encoder  (pretrained, fine-tuned)
• [CLS] representation per (question, option) pair
• Shared linear head  → scalar score per option
• Logits [B, 5]  fed to MCQLoss  (same loss as BiLSTM)

The model encodes all 5 options in parallel by flattening
[B, 5, L]  →  [B×5, L], running the encoder once,
then reshaping back to [B, 5].
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
    freeze_layers      : freeze the first N transformer layers (0 = train all)
    """

    def __init__(
        self,
        model_name         : str   = "microsoft/deberta-v3-base",
        classifier_dropout : float = 0.1,
        freeze_layers      : int   = 0,
    ):
        super().__init__()

        from transformers import AutoModel, AutoConfig

        hf_config                        = AutoConfig.from_pretrained(model_name)
        hf_config.hidden_dropout_prob    = classifier_dropout
        hf_config.attention_probs_dropout_prob = classifier_dropout

        self.encoder = AutoModel.from_pretrained(
            model_name, config=hf_config)

        hidden = hf_config.hidden_size          # 768 for -base, 1024 for -large

        # ── classifier head ──────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),          # scalar score per option
        )

        self._init_head()

        # ── optional layer freezing ───────────────────────────────────────────
        if freeze_layers > 0:
            self._freeze(freeze_layers)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MCQDeBERTa | trainable params: {n:,}")

    # ── weight init ──────────────────────────────────────────────────────────

    def _init_head(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ── freeze transformer layers ─────────────────────────────────────────────

    def _freeze(self, n_layers: int):
        """
        Freeze embedding + first n_layers of the transformer.
        Useful when GPU memory is tight or dataset is small.
        """
        # embeddings
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        # transformer layers  (attribute name differs by model family)
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

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ──────────
        input_ids      : [B, N_options, L]   (N_options = 5)
        attention_mask : [B, N_options, L]

        Returns
        ───────
        logits         : [B, N_options]      (raw scores, higher = more likely)
        """
        B, N, L = input_ids.shape

        # flatten options into batch dimension
        ids  = input_ids     .view(B * N, L)
        mask = attention_mask.view(B * N, L)

        # DeBERTa encoder → take [CLS] token (index 0)
        out  = self.encoder(input_ids=ids, attention_mask=mask)
        cls  = out.last_hidden_state[:, 0, :]   # [B*N, hidden]

        logits = self.head(cls).view(B, N)       # [B, N]
        return logits