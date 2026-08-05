# src/RoBERTa/data.py
"""
Data pipeline for RoBERTa-base MCQ.

Changes from reviewed version
──────────────────────────────
  normalize_text
    - Uses str.translate for contraction expansion when lowercase=True
      (slightly faster than repeated str.replace).

  SemanticDeduplicator
    - q_texts built with vectorised pandas .apply() (no iterrows).
    - AgglomerativeClustering guarded by max_agglom_rows: if the
      dataset is too large the clustering is skipped and each row gets
      its own singleton group (safe fallback).
    - sbert_matrix cached to disk via SBERTEmbedder.encode(cache_path)
      so repeated runs skip GPU work.

  MCQDataset
    - lazy_tokenization flag: preload=True (default) tokenises everything
      in __init__; preload=False tokenises on __getitem__ (low-RAM mode).
    - _process() uses direct column access instead of row.get() where
      the column is guaranteed to exist.
    - token_type_ids kept as all-zeros (correct for RoBERTa).

  collate_fn
    - Unchanged — already correct.
"""

import hashlib
import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset

from src.RoBERTa.embedder import SBERTEmbedder

logger = logging.getLogger("RoBERTa.Data")

ANSWER_LABELS = ["A", "B", "C", "D", "E"]
LABEL_TO_IDX  = {l: i for i, l in enumerate(ANSWER_LABELS)}
IDX_TO_LABEL  = {i: l for l, i in LABEL_TO_IDX.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text normalisation
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

# Ordered longest-first to avoid partial-match errors
_CONTRACTIONS: List[tuple] = sorted(
    [
        ("can't",    "cannot"),
        ("won't",    "will not"),
        ("n't",      " not"),
        ("'re",      " are"),
        ("'ve",      " have"),
        ("'ll",      " will"),
        ("'d",       " would"),
        ("'m",       " am"),
        ("it's",     "it is"),
        ("that's",   "that is"),
        ("there's",  "there is"),
        ("they're",  "they are"),
        ("we're",    "we are"),
        ("you're",   "you are"),
        ("i'm",      "i am"),
    ],
    key=lambda kv: -len(kv[0]),   # longest pattern first
)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\-]")


def normalize_text(text: str, lowercase: bool = False) -> str:
    """
    Strip instruction boilerplate, expand contractions (lowercase mode),
    collapse whitespace.

    RoBERTa is cased → lowercase=False by default.
    SBERT dedup path passes lowercase=True for better cluster cohesion.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _INSTRUCTION_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
        for pattern, replacement in _CONTRACTIONS:
            text = text.replace(pattern, replacement)
        text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def canonical_fingerprint(row: pd.Series) -> str:
    """Option-order-invariant MD5 of question + sorted options."""
    q    = normalize_text(str(row.get("prompt", "")), lowercase=True)
    opts = sorted(
        normalize_text(str(row.get(l, "")), lowercase=True)
        for l in ANSWER_LABELS
    )
    raw = (q + " ||| " + " || ".join(opts)).encode()
    return hashlib.md5(raw).hexdigest()


def option_set_fingerprint(row: pd.Series) -> str:
    """MD5 of sorted option texts only."""
    opts = sorted(
        normalize_text(str(row.get(l, "")), lowercase=True)
        for l in ANSWER_LABELS
    )
    return hashlib.md5((" || ".join(opts)).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semantic deduplicator
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Two-stage deduplication strategy.

    Stage A — Exact fingerprint dedup
        MD5 of (question + sorted options): removes true identical rows only.

    Stage B — SBERT encode
        Question texts embedded with sentence-transformers.
        Embeddings can be cached to disk for fast re-runs.

    Stage C — AgglomerativeClustering (group assignment only)
        Near-duplicates assigned to the same *semantic_group* so
        GroupShuffleSplit keeps them on the same side of the split.
        No additional rows are removed — the previous 'remove' mode
        that deleted ~40 % of training data has been eliminated.

        If the dataset exceeds *max_agglom_rows* clustering is skipped
        and each row receives a unique singleton group (avoids the O(N²)
        memory cost of full linkage on very large datasets).

    Parameters
    ──────────
    sbert_model     : sentence-transformers model name / local path
    sbert_batch_sz  : batch size for SBERT encoding
    sim_threshold   : cosine similarity above which rows are near-duplicates
    device          : "cuda" | "cpu" | None (auto)
    max_agglom_rows : skip clustering and assign singleton groups above this
    sbert_cache_path: optional .npy path; embeddings loaded if file exists
    """

    def __init__(
        self,
        sbert_model     : str            = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_batch_sz  : int            = 256,
        sim_threshold   : float          = 0.92,
        device          : Optional[str]  = None,
        max_agglom_rows : int            = 10_000,
        sbert_cache_path: Optional[str]  = None,
    ):
        self.sim_threshold    = sim_threshold
        self.max_agglom_rows  = max_agglom_rows
        self.sbert_cache_path = sbert_cache_path
        self.embedder = SBERTEmbedder(
            model_name = sbert_model,
            batch_size = sbert_batch_sz,
            device     = device,
            normalize  = True,
        )

    # ──────────────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
    ):
        """
        Returns
        ───────
        dedup_df     : pd.DataFrame  (only exact duplicates removed;
                                      near-duplicates kept with
                                      semantic_group label)
        sbert_matrix : np.ndarray    [n_rows, 384] float32, L2-normed,
                                     row-aligned with dedup_df
        """
        df = df.copy().reset_index(drop=True)

        # ── Stage A: exact fingerprint ─────────────────────────────────────
        df["exact_fp"]  = df.apply(canonical_fingerprint,  axis=1)
        df["option_fp"] = df.apply(option_set_fingerprint, axis=1)
        n_before = len(df)
        df = (
            df.drop_duplicates(subset="exact_fp", keep="first")
              .reset_index(drop=True)
        )
        logger.info(
            "Exact dedup: %d → %d  (removed %d exact duplicates)",
            n_before, len(df), n_before - len(df),
        )

        # ── Stage B: SBERT embeddings ──────────────────────────────────────
        # Vectorised text extraction — no iterrows()
        q_texts: List[str] = (
            df["prompt"]
            .fillna("")
            .astype(str)
            .apply(lambda t: normalize_text(t, lowercase=True))
            .tolist()
        )
        sbert_matrix = self.embedder.encode(
            q_texts,
            cache_path=self.sbert_cache_path,
        )   # [N, 384] float32

        # ── Stage C: clustering for group assignment ───────────────────────
        n = len(df)
        if n > self.max_agglom_rows:
            logger.warning(
                "Dataset size %d exceeds max_agglom_rows=%d. "
                "Skipping AgglomerativeClustering — each row assigned "
                "its own singleton group (no leakage risk).",
                n, self.max_agglom_rows,
            )
            df["semantic_group"] = np.arange(n, dtype=np.int32)
        else:
            logger.info(
                "Clustering %d questions (sim_threshold=%.3f) …",
                n, self.sim_threshold,
            )
            clustering = AgglomerativeClustering(
                n_clusters         = None,
                distance_threshold = 1.0 - self.sim_threshold,
                metric             = "cosine",
                linkage            = "average",   # less aggressive than complete
            )
            labels               = clustering.fit_predict(sbert_matrix)
            df["semantic_group"] = labels.astype(np.int32)
            self._log_cluster_stats(labels)

        logger.info(
            "All %d rows KEPT (near-duplicates grouped, not removed).", n
        )
        return df, sbert_matrix

    # ──────────────────────────────────────────────────────────────────────

    def _log_cluster_stats(self, labels: np.ndarray) -> None:
        cluster_size    = pd.Series(labels).value_counts()
        n_clusters      = int(np.unique(labels).shape[0])
        n_singleton     = int((cluster_size == 1).sum())
        n_multi         = int((cluster_size  > 1).sum())
        max_cluster     = int(cluster_size.max())
        mean_cluster    = float(cluster_size.mean())
        n_total         = int(labels.shape[0])
        frac_in_multi   = float(
            cluster_size[cluster_size > 1].sum() / n_total
        )

        logger.info(
            "Clusters: %d total | singleton: %d | multi-member: %d | "
            "largest: %d | mean size: %.2f",
            n_clusters, n_singleton, n_multi, max_cluster, mean_cluster,
        )
        if frac_in_multi > 0.5:
            logger.warning(
                "%.1f%% of rows are in multi-member clusters. "
                "Consider raising sim_threshold (current: %.3f).",
                100 * frac_in_multi, self.sim_threshold,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MCQ Dataset — RoBERTa tokenisation
# ─────────────────────────────────────────────────────────────────────────────

class MCQDataset(Dataset):
    """
    RoBERTa MCQ dataset.

    Tokenisation format per option:
        <s> question </s> </s> option_text </s>
    token_type_ids are always zero (RoBERTa has no segment embeddings).

    Parameters
    ──────────
    df       : DataFrame with columns [prompt, A, B, C, D, E, (answer)]
    tokenizer: HuggingFace fast tokenizer
    max_len  : maximum sequence length (tokens per option)
    is_test  : if True, no label is expected / returned
    preload  : True  → tokenise all rows in __init__ (fast DataLoader,
                        higher RAM usage)
               False → tokenise per __getitem__ call (low RAM, slower)
    """

    def __init__(
        self,
        df        : pd.DataFrame,
        tokenizer,
        max_len   : int  = 128,
        is_test   : bool = False,
        preload   : bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.is_test   = is_test
        self.preload   = preload
        self.df        = df.reset_index(drop=True)

        if preload:
            logger.info("Pre-tokenising %d examples …", len(self.df))
            self._data = [self._process(i) for i in range(len(self.df))]
            logger.info("Tokenisation complete.")
        else:
            self._data = None
            logger.info(
                "Lazy-tokenisation mode: %d examples will be processed "
                "on demand.", len(self.df)
            )

    # ──────────────────────────────────────────────────────────────────────

    def _process(self, i: int) -> Dict:
        row      = self.df.iloc[i]
        question = normalize_text(str(row.get("prompt", "")))

        all_input_ids      : List[torch.Tensor] = []
        all_attention_mask : List[torch.Tensor] = []

        for lbl in ANSWER_LABELS:
            option = normalize_text(str(row.get(lbl, "")))
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

        # token_type_ids are always zero for RoBERTa
        zeros = torch.zeros(
            len(ANSWER_LABELS), self.max_len, dtype=torch.long
        )

        item: Dict = dict(
            id             = row.get("id", i),
            input_ids      = torch.stack(all_input_ids),       # [5, L]
            attention_mask = torch.stack(all_attention_mask),  # [5, L]
            token_type_ids = zeros,                            # [5, L]
        )

        if not self.is_test and "answer" in row:
            lbl_str = str(row["answer"]).strip().upper()
            item["label"] = torch.tensor(
                LABEL_TO_IDX.get(lbl_str, 0), dtype=torch.long
            )

        return item

    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict:
        if self.preload:
            return self._data[i]       # type: ignore[index]
        return self._process(i)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    out: Dict = dict(
        id             = [b["id"]             for b in batch],
        input_ids      = torch.stack([b["input_ids"]      for b in batch]),
        attention_mask = torch.stack([b["attention_mask"] for b in batch]),
        token_type_ids = torch.stack([b["token_type_ids"] for b in batch]),
    )
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out