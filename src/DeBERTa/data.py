# src/DeBERTa/data.py
"""
Data utilities for DeBERTa-v3.

Key differences from BiLSTM:
  • Uses HuggingFace tokenizer instead of a hand-built Vocabulary
  • Each (question, option) pair is encoded as a single [CLS] ... [SEP] sequence
  • Returns input_ids, attention_mask, token_type_ids  (no lengths needed)
  • Deduplication / fingerprinting / BoW audit are IDENTICAL to BiLSTM
"""

import logging
from torch.utils.data import Dataset
import torch

# ── reuse BiLSTM helpers verbatim ────────────────────────────────────────────
from src.BiLSTM.data import (
    normalize_text,
    canonical_fingerprint,
    option_set_fingerprint,
    BagOfWordsEncoder,
    SemanticDeduplicator,       # same BoW + Agglomerative pipeline
    ANSWER_LABELS,
    LABEL_TO_IDX,
)

logger = logging.getLogger("DeBERTa.Data")


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer factory
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str):
    """
    Returns a DeBERTa-v3 tokenizer.
    Uses AutoTokenizer so the same call works for -small / -base / -large.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer loaded: {model_name}  "
                f"(vocab={tok.vocab_size:,})")
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MCQDataset(Dataset):
    """
    One sample  →  5 encoded pairs (question+optionA … question+optionE).

    Encoding template per pair
    ──────────────────────────
    [CLS] <question_text> [SEP] <option_text> [SEP]

    This is the standard single-sentence-pair format for DeBERTa / RoBERTa.
    token_type_ids are zeros throughout (DeBERTa-v3 ignores segment IDs).
    """

    def __init__(self, df, tokenizer, max_len: int = 256,
                 is_test: bool = False):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.is_test   = is_test
        self.data      = [self._process(i) for i in range(len(self.df))]

    def _process(self, i: int) -> dict:
        row = self.df.iloc[i]
        q   = normalize_text(str(row.get('prompt', '')))

        input_ids_list      = []
        attention_mask_list = []

        for lbl in ANSWER_LABELS:
            opt = normalize_text(str(row.get(lbl, '')))

            # HuggingFace handles [CLS]/[SEP] insertion automatically
            enc = self.tokenizer(
                q,                          # text_a
                opt,                        # text_b
                max_length      = self.max_len,
                padding         = 'max_length',
                truncation      = True,
                return_tensors  = 'pt',
            )
            # squeeze the batch dim added by return_tensors='pt'
            input_ids_list     .append(enc['input_ids'].squeeze(0))
            attention_mask_list.append(enc['attention_mask'].squeeze(0))

        item = dict(
            id             = row.get('id', i),
            # shape: [5, max_len]
            input_ids      = torch.stack(input_ids_list),
            attention_mask = torch.stack(attention_mask_list),
        )

        if not self.is_test and 'answer' in row:
            lbl = str(row['answer']).strip().upper()
            item['label'] = torch.tensor(
                LABEL_TO_IDX.get(lbl, 0), dtype=torch.long)

        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Stacks tensors; keeps id as a plain list (strings or ints).
    """
    out = dict(
        id             = [b['id']             for b in batch],
        input_ids      = torch.stack([b['input_ids']       for b in batch]),
        attention_mask = torch.stack([b['attention_mask']  for b in batch]),
    )
    if 'label' in batch[0]:
        out['label'] = torch.stack([b['label'] for b in batch])
    return out