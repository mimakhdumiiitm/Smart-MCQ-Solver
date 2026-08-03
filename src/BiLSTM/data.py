# data.py
"""
All data logic in one file:
  - Text normalization
  - Fingerprinting
  - Semantic deduplication
  - Vocabulary
  - Dataset + collate_fn
"""

import re
import hashlib
import pickle
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger("Data")

ANSWER_LABELS = ['A', 'B', 'C', 'D', 'E']
LABEL_TO_IDX  = {l: i for i, l in enumerate(ANSWER_LABELS)}

# ─────────────────────────────────────────────────────────────────────────────
# Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

_INSTRUCTION_RE = re.compile(
    r"""
    (pick\s+the\s+best(\s+possible)?\s+answer\s*[:\-]?\s*)     |
    (determine\s+the(\s+correct)?\s+option\s*[:\-]?\s*)        |
    (select\s+the(\s+most\s+accurate)?\s+option\s*[:\-]?\s*)   |
    (choose\s+the(\s+correct)?\s+answer\s*[:\-]?\s*)           |
    (identify\s+the(\s+best)?\s+answer\s*[:\-]?\s*)            |
    (what\s+is\s+the\s+best\s+answer\s*[:\-]?\s*)              |
    (which\s+of\s+the\s+following\s+is\s+correct\s*[:\-]?\s*)  |
    (among\s+the\s+(listed\s+)?options\.?\s*)                   |
    (from\s+the(\s+options)?\s+given\.?\s*)                     |
    (answer\s+the\s+following(\s+question)?\s*[:\-]?\s*)        |
    (the\s+correct\s+answer\s+is\s*[:\-]?\s*)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CONTRACTIONS = {
    "can't": "cannot",    "won't": "will not",   "n't": " not",
    "'re":   " are",      "'ve":   " have",       "'ll": " will",
    "'d":    " would",    "'m":    " am",          "it's": "it is",
    "that's":"that is",   "there's":"there is",   "they're":"they are",
    "we're": "we are",    "you're":"you are",     "i'm":  "i am",
    "he's":  "he is",     "she's": "she is",      "isn't":"is not",
    "aren't":"are not",   "wasn't":"was not",     "weren't":"were not",
    "doesn't":"does not", "don't": "do not",      "didn't":"did not",
    "hasn't":"has not",   "haven't":"have not",   "hadn't":"had not",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _INSTRUCTION_RE.sub(' ', text)
    for c, e in _CONTRACTIONS.items():
        text = text.replace(c, e)
    text = re.sub(r'[""\'`´]',      '',  text)
    text = re.sub(r'[\[\]\(\)\{\}]',' ', text)
    text = text.replace('–', '-').replace('—', '-').replace('…', '...')
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+',       ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def canonical_fingerprint(row: pd.Series) -> str:
    """Option-order-invariant MD5 of question + sorted options."""
    q    = normalize_text(str(row.get('prompt', '')))
    opts = sorted(normalize_text(str(row.get(l, ''))) for l in ANSWER_LABELS)
    return hashlib.md5((q + ' ||| ' + ' || '.join(opts)).encode()).hexdigest()


def option_set_fingerprint(row: pd.Series) -> str:
    """MD5 of sorted option texts only — catches same options, different question."""
    opts = sorted(normalize_text(str(row.get(l, ''))) for l in ANSWER_LABELS)
    return hashlib.md5((' || '.join(opts)).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Deduplicator
# ─────────────────────────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False


class SemanticDeduplicator:
    """
    Stage A — exact option-order-invariant fingerprint dedup
    Stage B — SBERT cosine + AgglomerativeClustering
    Returns df with [exact_fp, option_fp, semantic_group] + embeddings array
    """

    def __init__(self, sbert_model='all-MiniLM-L6-v2',
                 sim_threshold=0.85, batch_size=128):
        self.sim_threshold = sim_threshold
        self.use_sbert     = _SBERT_OK

        if self.use_sbert:
            logger.info(f"Loading SBERT: {sbert_model}")
            self.sbert = SentenceTransformer(sbert_model)
        else:
            logger.warning("SBERT unavailable → TF-IDF fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(
                ngram_range=(1,2), max_features=30_000, sublinear_tf=True)
        self.batch_size = batch_size

    def _encode(self, texts):
        if self.use_sbert:
            embs = self.sbert.encode(
                texts, batch_size=self.batch_size,
                show_progress_bar=True, normalize_embeddings=True)
            return embs.astype(np.float32)
        mat   = self.tfidf.fit_transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy().reset_index(drop=True)

        # Stage A — exact dedup
        df['exact_fp']  = df.apply(canonical_fingerprint,  axis=1)
        df['option_fp'] = df.apply(option_set_fingerprint, axis=1)
        n_before = len(df)
        df = df.drop_duplicates(subset='exact_fp', keep='first').reset_index(drop=True)
        logger.info(f"Exact dedup: {n_before} → {len(df)} "
                    f"(removed {n_before - len(df)})")

        # Stage B — semantic clustering
        q_texts = [normalize_text(str(r.get('prompt','')))
                   for _, r in df.iterrows()]
        embs    = self._encode(q_texts)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.sim_threshold,
            metric='cosine', linkage='average')
        df['semantic_group'] = clustering.fit_predict(embs)

        n_clusters = len(np.unique(df['semantic_group']))
        logger.info(f"Semantic clusters: {n_clusters:,}  "
                    f"multi-row: {(pd.Series(df['semantic_group']).value_counts()>1).sum():,}")
        return df, embs


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD, UNK, SEP = '<PAD>', '<UNK>', '[SEP]'
    PAD_IDX = 0; UNK_IDX = 1; SEP_IDX = 2

    def __init__(self, max_vocab=20_000, min_freq=2):
        self.max_vocab = max_vocab
        self.min_freq  = min_freq
        self.word2idx  = {}
        self.idx2word  = {}
        self._freq     = Counter()

    def build(self, texts):
        for t in texts:
            self._freq.update(t.split())
        self.word2idx = {self.PAD: 0, self.UNK: 1, self.SEP: 2}
        idx = 3
        for word, freq in self._freq.most_common(self.max_vocab):
            if freq < self.min_freq:
                break
            if word not in self.word2idx:
                self.word2idx[word] = idx; idx += 1
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        logger.info(f"Vocab size: {len(self.word2idx):,}")
        return self

    def encode(self, text: str, max_len: int = None):
        toks = text.split()
        if max_len:
            toks = toks[:max_len]
        return [self.word2idx.get(t, self.UNK_IDX) for t in toks]

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MCQDataset(Dataset):

    def __init__(self, df, vocab, max_len=180, is_test=False):
        self.vocab   = vocab
        self.max_len = max_len
        self.is_test = is_test
        self.df      = df.reset_index(drop=True)
        self.data    = [self._process(i) for i in range(len(self.df))]

    def _process(self, i):
        row = self.df.iloc[i]
        q   = normalize_text(str(row.get('prompt', '')))

        seqs, lens = [], []
        for lbl in ANSWER_LABELS:
            text = f"{q} {self.vocab.SEP} {normalize_text(str(row.get(lbl, '')))}"
            ids  = self.vocab.encode(text, self.max_len)
            lens.append(max(len(ids), 1))
            ids += [self.vocab.PAD_IDX] * (self.max_len - len(ids))
            seqs.append(ids)

        item = dict(
            id      = row.get('id', i),
            options = torch.tensor(seqs,  dtype=torch.long),
            lengths = torch.tensor(lens,  dtype=torch.long),
        )
        if not self.is_test and 'answer' in row:
            lbl = str(row['answer']).strip().upper()
            item['label'] = torch.tensor(LABEL_TO_IDX.get(lbl, 0),
                                         dtype=torch.long)
        return item

    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    out = dict(
        id      = [b['id']      for b in batch],
        options = torch.stack([b['options'] for b in batch]),
        lengths = torch.stack([b['lengths'] for b in batch]),
    )
    if 'label' in batch[0]:
        out['label'] = torch.stack([b['label'] for b in batch])
    return out