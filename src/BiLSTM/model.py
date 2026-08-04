# model.py
"""
Bi-LSTM MCQ model — fully from scratch, no pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _ScaledDotAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj  = nn.Linear(hidden * 2, hidden * 2, bias=False)
        self.scale = (hidden * 2) ** 0.5

    def forward(self, h, mask=None):
        q      = self.proj(h[:, -1:, :])
        scores = torch.bmm(q, h.transpose(1, 2)).squeeze(1) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        w = F.softmax(scores, dim=-1)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1), w


class _BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, n_layers, dropout, pad_idx):
        super().__init__()
        self.emb      = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm     = nn.LSTM(
            embed_dim, hidden,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.,
        )
        self.attn = _ScaledDotAttention(hidden)
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        emb         = self.emb_drop(self.emb(x))
        packed      = pack_padded_sequence(
            emb, lengths.clamp(min=1).cpu(),
            batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _      = pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))
        ctx, _      = self.attn(out, mask=(x == 0))
        final       = torch.cat([h[-2], h[-1]], dim=-1)
        return self.drop(self.norm(ctx + final))


class MCQBiLSTM(nn.Module):
    """
    Scores each (question, option) pair independently,
    returns logits [B, N_options].
    All weights randomly initialised — no pretrained components.
    """

    def __init__(self, vocab_size, embed_dim=100, hidden=128,
                 n_layers=2, dropout=0.4, pad_idx=0):
        super().__init__()
        self.encoder = _BiLSTMEncoder(
            vocab_size, embed_dim, hidden, n_layers, dropout, pad_idx)
        feat = hidden * 2
        self.scorer = nn.Sequential(
            nn.Linear(feat, feat // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(feat // 2),
            nn.Linear(feat // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and 'lstm' in name:
                nn.init.orthogonal_(p)
            elif 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.encoder.emb.weight, std=0.01)
        nn.init.zeros_(self.encoder.emb.weight[0])

    def forward(self, options, lengths):
        B, N, L = options.shape
        rep     = self.encoder(options.view(B * N, L), lengths.view(B * N))
        return self.scorer(rep).view(B, N)