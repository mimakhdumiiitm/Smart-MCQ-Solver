# src/RoBERTa/data.py
"""
Data pipeline for RoBERTa-base MCQ.

Key fixes vs previous version
──────────────────────────────
  SemanticDeduplicator
    - sim_threshold default raised to 0.92  (less aggressive)
    - Added dedup_mode: 'group_only' keeps ALL rows but assigns
      semantic_group for clean splitting; no rows are removed by
      clustering — only exact duplicates are dropped.
      This is the critical fix: the previous version lost 40% of
      training data by treating near-duplicates as true duplicates.
    - Detailed logging of cluster size distribution

  MCQDataset
    - token_type_ids always zeros for RoBERTa (unchanged, correct)
    - normalize_text keeps casing (RoBERTa is cased)
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

from src.RoBERTa.embedder import SBERTEmbedder

logger = logging.getLogger("RoBERTa.Data")

ANSWER_LABELS = ['A', 'B', 'C', 'D', 'E']
LABEL_TO_IDX  = {l: i for i, l in enumerate(ANSWER_LABELS)}
IDX_TO_LABEL  = {i: l for l, i in LABEL_TO_IDX.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Normalization
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
    SBERT dedup uses lowercase=True for better matching.
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
    """Option-order-invariant MD5 of question + sorted options."""
    q    = normalize_text(str(row.get('prompt', '')), lowercase=True)
    opts = sorted(
        normalize_text(str(row.get(l, '')), lowercase=True)
        for l in ANSWER_LABELS
    )
    return hashlib.md5(
        (q + ' ||| ' + ' || '.join(opts)).encode()
    ).hexdigest()


def option_set_fingerprint(row: pd.Series) -> str:
    """MD5 of sorted option texts only."""
    opts = sorted(
        normalize_text(str(row.get(l, '')), lowercase=True)
        for l in ANSWER_LABELS
    )
    return hashlib.md5((' || '.join(opts)).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semantic Deduplicator
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Two-stage deduplication:

    Stage A  Exact fingerprint dedup  (MD5 of Q + sorted options)
             → removes true identical questions only

    Stage B  SBERT encode + AgglomerativeClustering
             → assigns semantic_group labels for GroupShuffleSplit
             → does NOT remove any additional rows
             → near-duplicates stay in training but are kept in the
                same group so they cannot leak across the split boundary

    Why 'group_only' mode
    ─────────────────────
    The previous 'remove' mode deleted ~40% of training data by treating
    near-duplicates as redundant. But near-duplicates with different correct
    answers (same question, rephrased) are valuable training signal.
    GroupShuffleSplit already prevents leakage by keeping the whole cluster
    on one side of the split — removing them is unnecessary and harmful.

    Parameters
    ──────────
    sbert_model    : sentence-transformers model name
    sbert_batch_sz : batch size for SBERT encoding
    sim_threshold  : cosine similarity above which Qs are near-duplicates
                     0.92 is less aggressive than previous 0.85
    device         : "cuda" | "cpu"
    """

    def __init__(
        self,
        sbert_model    : str   = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_batch_sz : int   = 256,
        sim_threshold  : float = 0.92,
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
        dedup_df     : pd.DataFrame  with exact_fp, option_fp,
                                     semantic_group columns
                                     (only exact duplicates removed)
        sbert_matrix : np.ndarray    [n_rows, 384] L2-normed SBERT vectors,
                                     aligned row-for-row with dedup_df
        """
        df = df.copy().reset_index(drop=True)

        # ── Stage A: exact fingerprint (true duplicates only) ─────────────
        df['exact_fp']  = df.apply(canonical_fingerprint,  axis=1)
        df['option_fp'] = df.apply(option_set_fingerprint, axis=1)
        n_before = len(df)
        df = (df.drop_duplicates(subset='exact_fp', keep='first')
                .reset_index(drop=True))
        logger.info(
            f"Exact dedup: {n_before} → {len(df)} "
            f"(removed {n_before - len(df):,} exact duplicates)"
        )

        # ── Stage B: SBERT embeddings ──────────────────────────────────────
        q_texts = [
            normalize_text(str(r.get('prompt', '')), lowercase=True)
            for _, r in df.iterrows()
        ]
        sbert_matrix = self.embedder.encode(q_texts)   # [N, 384] float32

        # ── Stage C: clustering for group assignment only ──────────────────
        logger.info(
            f"Clustering {len(df):,} questions "
            f"(sim_threshold={self.sim_threshold}) …"
        )
        clustering = AgglomerativeClustering(
            n_clusters         = None,
            distance_threshold = 1.0 - self.sim_threshold,
            metric             = 'cosine',
            linkage            = 'average',    # average linkage: less aggressive
        )                                      # than complete linkage
        labels               = clustering.fit_predict(sbert_matrix)
        df['semantic_group'] = labels

        # ── Diagnostic logging ────────────────────────────────────────────
        n_clusters   = int(np.unique(labels).shape[0])
        cluster_size = pd.Series(labels).value_counts()
        n_singleton  = int((cluster_size == 1).sum())
        n_multi      = int((cluster_size  > 1).sum())
        max_cluster  = int(cluster_size.max())
        mean_cluster = float(cluster_size.mean())

        logger.info(
            f"Clusters: {n_clusters:,} total | "
            f"singleton: {n_singleton:,} | "
            f"multi-member: {n_multi:,} | "
            f"largest: {max_cluster} | "
            f"mean size: {mean_cluster:.2f}"
        )
        logger.info(
            f"All {len(df):,} rows KEPT "
            f"(near-duplicates grouped, not removed)"
        )

        # Warn if clustering is suspiciously aggressive
        fraction_in_multi = float(
            cluster_size[cluster_size > 1].sum() / len(df)
        )
        if fraction_in_multi > 0.5:
            logger.warning(
                f"{100*fraction_in_multi:.1f}% of rows are in multi-member "
                f"clusters — consider raising sim_threshold "
                f"(current: {self.sim_threshold})."
            )

        return df, sbert_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 4. MCQ Dataset — RoBERTa tokenization
# ─────────────────────────────────────────────────────────────────────────────

class MCQDataset(Dataset):
    """
    Pre-tokenizes each (question, option) pair.

    RoBERTa format per option:
        <s> question </s> </s> option_text </s>

    token_type_ids are always zero for RoBERTa.
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

        logger.info(f"Tokenizing {len(self.df):,} examples …")
        self.data = [self._process(i) for i in range(len(self.df))]
        logger.info("Tokenization complete.")

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
                max_length     = self.max_len,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt",
            )
            all_input_ids.append(enc["input_ids"].squeeze(0))
            all_attention_mask.append(enc["attention_mask"].squeeze(0))
            all_token_type_ids.append(
                torch.zeros(self.max_len, dtype=torch.long)
            )

        item = dict(
            id             = row.get('id', i),
            input_ids      = torch.stack(all_input_ids),
            attention_mask = torch.stack(all_attention_mask),
            token_type_ids = torch.stack(all_token_type_ids),
        )

        if not self.is_test and 'answer' in row:
            lbl_str = str(row['answer']).strip().upper()
            item['label'] = torch.tensor(
                LABEL_TO_IDX.get(lbl_str, 0), dtype=torch.long)

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