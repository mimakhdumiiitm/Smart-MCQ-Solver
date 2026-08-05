# src/RoBERTa/data.py
"""
Data pipeline for RoBERTa-base MCQ.

Key differences from DeBERTa version
──────────────────────────────────────
  - RoBERTa has NO token_type_ids  → we return zeros (model ignores them)
  - normalize_text keeps casing    (RoBERTa is cased)
  - SemanticDeduplicator identical (SBERT-based)
"""

import re
import hashlib
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.RoBERTa.embedder import SBERTEmbedder

logger = logging.getLogger("RoBERTa.Data")

ANSWER_LABELS = ['A', 'B', 'C', 'D', 'E']
LABEL_TO_IDX  = {l: i for i, l in enumerate(ANSWER_LABELS)}
IDX_TO_LABEL  = {i: l for l, i in LABEL_TO_IDX.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Normalisation
# ─────────────────────────────────────────────────────────────────────────────

_INSTRUCTION_RE = re.compile(
    r"""
    (pick\s+the\s+best(\s+possible)?\s+answer\s*[:\-]?\s*)      |
    (determine\s+the(\s+correct)?\s+option\s*[:\-]?\s*)         |
    (select\s+the(\s+most\s+accurate)?\s+option\s*[:\-]?\s*)    |
    (choose\s+the(\s+correct)?\s+answer\s*[:\-]?\s*)            |
    (identify\s+the(\s+best)?\s+answer\s*[:\-]?\s*)             |
    (what\s+is\s+the\s+best\s+answer\s*[:\-]?\s*)               |
    (which\s+of\s+the\s+following\s+is\s+correct\s*[:\-]?\s*)   |
    (among\s+the\s+(listed\s+)?options\.?\s*)                    |
    (from\s+the(\s+options)?\s+given\.?\s*)                      |
    (answer\s+the\s+following(\s+question)?\s*[:\-]?\s*)         |
    (the\s+correct\s+answer\s+is\s*[:\-]?\s*)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CONTRACTIONS = {
    "can't": "cannot",      "won't": "will not",
    "n't":   " not",        "'re":   " are",
    "'ve":   " have",       "'ll":   " will",
    "'d":    " would",      "'m":    " am",
    "it's":  "it is",       "that's":"that is",
    "there's":"there is",   "they're":"they are",
    "we're": "we are",      "you're":"you are",
    "i'm":   "i am",
}


def normalize_text(text: str, lowercase: bool = False) -> str:
    """
    Strip instruction boilerplate, expand contractions, clean whitespace.
    RoBERTa is cased → lowercase=False by default.
    SBERT dedup uses lowercase=True.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _INSTRUCTION_RE.sub(' ', text)
    if lowercase:
        text = text.lower()
        for c, e in _CONTRACTIONS.items():
            text = text.replace(c, e)
        text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def canonical_fingerprint(row: pd.Series) -> str:
    q    = normalize_text(str(row.get('prompt', '')), lowercase=True)
    opts = sorted(
        normalize_text(str(row.get(l, '')), lowercase=True)
        for l in ANSWER_LABELS
    )
    return hashlib.md5(
        (q + ' ||| ' + ' || '.join(opts)).encode()
    ).hexdigest()


def option_set_fingerprint(row: pd.Series) -> str:
    opts = sorted(
        normalize_text(str(row.get(l, '')), lowercase=True)
        for l in ANSWER_LABELS
    )
    return hashlib.md5((' || '.join(opts)).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semantic Deduplicator — SBERT + AgglomerativeClustering
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Stage A : Exact MD5 fingerprint dedup
    Stage B : SBERT encode → 384-dim L2-normed vectors
    Stage C : AgglomerativeClustering → semantic_group for GroupShuffleSplit
    """

    def __init__(
        self,
        sbert_model    : str   = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_batch_sz : int   = 256,
        sim_threshold  : float = 0.85,
        device         : str   = None,
    ):
        self.sim_threshold = sim_threshold
        self.embedder = SBERTEmbedder(
            model_name = sbert_model,
            batch_size = sbert_batch_sz,
            device     = device,
            normalize  = True,
        )

    def fit_transform(self, df: pd.DataFrame):
        """
        Returns
        ───────
        dedup_df     : pd.DataFrame with exact_fp, option_fp, semantic_group
        sbert_matrix : np.ndarray [n_rows, 384] L2-normed
        """
        df = df.copy().reset_index(drop=True)

        # Stage A: exact fingerprint
        df['exact_fp']  = df.apply(canonical_fingerprint,  axis=1)
        df['option_fp'] = df.apply(option_set_fingerprint, axis=1)
        n_before = len(df)
        df = (df.drop_duplicates(subset='exact_fp', keep='first')
                .reset_index(drop=True))
        logger.info(f"Exact dedup: {n_before} → {len(df)} "
                    f"(removed {n_before - len(df):,})")

        # Stage B: SBERT embeddings
        q_texts = [
            normalize_text(str(r.get('prompt', '')), lowercase=True)
            for _, r in df.iterrows()
        ]
        sbert_matrix = self.embedder.encode(q_texts)   # [N, 384]

        # Stage C: AgglomerativeClustering
        logger.info(
            f"Clustering {len(df):,} questions "
            f"(sim_threshold={self.sim_threshold}) …"
        )
        clustering = AgglomerativeClustering(
            n_clusters         = None,
            distance_threshold = 1.0 - self.sim_threshold,
            metric             = 'cosine',
            linkage            = 'complete',
        )
        labels               = clustering.fit_predict(sbert_matrix)
        df['semantic_group'] = labels

        n_clusters   = len(np.unique(labels))
        cluster_size = pd.Series(labels).value_counts()
        n_removed    = (cluster_size - 1).clip(lower=0).sum()
        n_multi      = (cluster_size > 1).sum()
        logger.info(
            f"Clusters: {n_clusters:,}  "
            f"multi-member (near-dupes): {n_multi:,}  "
            f"rows that WOULD be removed at dedup: {n_removed:,}  "
            f"({100 * n_removed / len(df):.1f}%)"
        )
        if n_removed / len(df) > 0.25:
            logger.warning(
                f"Dedup would remove {100 * n_removed / len(df):.1f}% of data. "
                f"Consider raising sim_threshold."
            )
        return df, sbert_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 4. MCQ Dataset — RoBERTa tokenization
# ─────────────────────────────────────────────────────────────────────────────

class MCQDataset(Dataset):
    """
    Pre-tokenises each (question, option) pair.

    RoBERTa input:
        <s> question </s> </s> option </s>
    token_type_ids are all zeros (RoBERTa ignores segment ids).

    Output per sample:
        input_ids      : [5, max_len]
        attention_mask : [5, max_len]
        token_type_ids : [5, max_len]   ← zeros, kept for API compatibility
        label          : int  (absent for test)
        id             : any
    """

    def __init__(
        self,
        df        : pd.DataFrame,
        tokenizer,
        max_len   : int  = 128,
        is_test   : bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.is_test   = is_test
        self.df        = df.reset_index(drop=True)

        logger.info(f"Tokenising {len(self.df):,} examples …")
        self.data = [self._process(i) for i in range(len(self.df))]
        logger.info("Tokenisation complete.")

    def _process(self, i: int) -> Dict:
        row      = self.df.iloc[i]
        question = normalize_text(str(row.get('prompt', '')))

        all_input_ids      = []
        all_attention_mask = []
        all_token_type_ids = []

        for lbl in ANSWER_LABELS:
            option = normalize_text(str(row.get(lbl, '')))
            enc    = self.tokenizer(
                question,
                option,
                max_length      = self.max_len,
                padding         = "max_length",
                truncation      = True,
                return_tensors  = "pt",
            )
            all_input_ids.append(enc["input_ids"].squeeze(0))
            all_attention_mask.append(enc["attention_mask"].squeeze(0))

            # RoBERTa does NOT produce token_type_ids → use zeros
            if "token_type_ids" in enc:
                ttids = enc["token_type_ids"].squeeze(0)
            else:
                ttids = torch.zeros(self.max_len, dtype=torch.long)
            all_token_type_ids.append(ttids)

        item = dict(
            id             = row.get('id', i),
            input_ids      = torch.stack(all_input_ids),       # [5, L]
            attention_mask = torch.stack(all_attention_mask),  # [5, L]
            token_type_ids = torch.stack(all_token_type_ids),  # [5, L] zeros
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


def collate_fn(batch: List[Dict]) -> Dict:
    out = dict(
        id             = [b['id']             for b in batch],
        input_ids      = torch.stack([b['input_ids']      for b in batch]),
        attention_mask = torch.stack([b['attention_mask'] for b in batch]),
        token_type_ids = torch.stack([b['token_type_ids'] for b in batch]),
    )
    if 'label' in batch[0]:
        out['label'] = torch.stack([b['label'] for b in batch])
    return out