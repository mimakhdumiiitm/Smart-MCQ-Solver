# C:\aa all photos\all coding stuff\DL GenAI Project\Smart-MCQ-Solver\src\BiLSTM\data.py


import re
import hashlib
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
# 1.  Text Normalization
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
    "can't":"cannot",     "won't":"will not",    "n't":" not",
    "'re":" are",         "'ve":" have",          "'ll":" will",
    "'d":" would",        "'m":" am",             "it's":"it is",
    "that's":"that is",   "there's":"there is",   "they're":"they are",
    "we're":"we are",     "you're":"you are",     "i'm":"i am",
    "he's":"he is",       "she's":"she is",       "isn't":"is not",
    "aren't":"are not",   "wasn't":"was not",     "weren't":"were not",
    "doesn't":"does not", "don't":"do not",       "didn't":"did not",
    "hasn't":"has not",   "haven't":"have not",   "hadn't":"had not",
}


def normalize_text(text: str) -> str:
    """
    Lowercase → strip instruction wrappers → expand contractions →
    remove special chars → collapse whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _INSTRUCTION_RE.sub(' ', text)
    for c, e in _CONTRACTIONS.items():
        text = text.replace(c, e)
    text = re.sub(r'[""\'`´]',       '',  text)
    text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
    text = text.replace('–', '-').replace('—', '-').replace('…', '...')
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+',       ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def canonical_fingerprint(row: pd.Series) -> str:
    """Option-order-invariant MD5 of question + sorted options."""
    q    = normalize_text(str(row.get('prompt', '')))
    opts = sorted(normalize_text(str(row.get(l, ''))) for l in ANSWER_LABELS)
    return hashlib.md5((q + ' ||| ' + ' || '.join(opts)).encode()).hexdigest()


def option_set_fingerprint(row: pd.Series) -> str:
    """MD5 of sorted option texts only."""
    opts = sorted(normalize_text(str(row.get(l, ''))) for l in ANSWER_LABELS)
    return hashlib.md5((' || '.join(opts)).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Bag-of-Words Encoder  (from scratch — no sklearn TfidfVectorizer)
# ─────────────────────────────────────────────────────────────────────────────

class BagOfWordsEncoder:
    """
    Builds its own unigram/bigram vocabulary from the corpus,
    computes raw term-frequency vectors, applies IDF weighting,
    and L2-normalises so that dot-product == cosine similarity.

    Everything is pure NumPy — no pretrained weights anywhere.

    Workflow
    ────────
    fit(texts)          → build token vocab, compute IDF
    transform(texts)    → return float32 matrix [N, vocab_size], L2-normed
    fit_transform(texts)→ fit then transform
    """

    def __init__(self, max_features: int = 30_000, ngram_max: int = 2):
        self.max_features = max_features
        self.ngram_max    = ngram_max
        self.token2idx_   : dict  = {}
        self.idf_         : np.ndarray = None
        self._fitted      : bool  = False

    # ── tokenisation ─────────────────────────────────────────────────────────
    def _tokenize(self, text: str):
        tokens = text.split()                       # already normalised
        grams  = list(tokens)                       # unigrams
        if self.ngram_max >= 2:
            grams += [f"{a}_{b}"
                      for a, b in zip(tokens, tokens[1:])]   # bigrams
        return grams

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, texts):
        """
        Build vocabulary from corpus.
        Keep top-max_features tokens by document frequency.
        Compute smoothed IDF = log((1 + N) / (1 + df)) + 1
        """
        N          = len(texts)
        doc_freq   = Counter()

        for text in texts:
            grams = set(self._tokenize(text))   # unique per doc for IDF
            doc_freq.update(grams)

        # keep top-max_features by document frequency
        top_tokens = [tok for tok, _ in
                      doc_freq.most_common(self.max_features)]

        self.token2idx_ = {tok: i for i, tok in enumerate(top_tokens)}
        V = len(self.token2idx_)

        # IDF vector
        df_arr     = np.array([doc_freq[tok] for tok in top_tokens],
                               dtype=np.float32)
        self.idf_  = np.log((1.0 + N) / (1.0 + df_arr)) + 1.0
        self._fitted = True
        logger.info(f"BoW vocab: {V:,} tokens  (ngram_max={self.ngram_max})")
        return self

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(self, texts) -> np.ndarray:
        """
        Returns float32 matrix [len(texts), vocab_size].
        Each row is TF-IDF weighted and L2-normalised.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        V   = len(self.token2idx_)
        N   = len(texts)
        mat = np.zeros((N, V), dtype=np.float32)

        for i, text in enumerate(texts):
            grams = self._tokenize(text)
            # raw term frequency
            tf = Counter(grams)
            for gram, cnt in tf.items():
                j = self.token2idx_.get(gram)
                if j is not None:
                    mat[i, j] = cnt

            # apply IDF
            mat[i] *= self.idf_

        # L2 normalise each row  →  dot product == cosine similarity
        norms         = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0          # avoid division by zero
        mat          /= norms
        return mat

    def fit_transform(self, texts) -> np.ndarray:
        return self.fit(texts).transform(texts)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Semantic Deduplicator — BoW + own cosine + AgglomerativeClustering
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Deduplication pipeline — zero pretrained models:

      Stage A  exact option-order-invariant fingerprint dedup
      Stage B  BoW TF-IDF vectors  →  L2-normalise
               AgglomerativeClustering with our cosine distance (1 - cosine)
               assigns each question a semantic_group id
               used later for GroupShuffleSplit

    Parameters
    ──────────
    sim_threshold   : float  cosine similarity above which two questions
                             are considered near-duplicates (same cluster)
    max_features    : int    BoW vocabulary size
    ngram_max       : int    1 = unigrams, 2 = unigrams + bigrams
    """

    def __init__(
        self,
        sim_threshold : float = 0.85,
        max_features  : int   = 30_000,
        ngram_max     : int   = 2,
    ):
        self.sim_threshold = sim_threshold
        self.bow           = BagOfWordsEncoder(
            max_features=max_features, ngram_max=ngram_max)

    def fit_transform(self, df: pd.DataFrame):
        """
        Returns
        ───────
        dedup_df   : pd.DataFrame  with columns exact_fp, option_fp,
                                   semantic_group added
        bow_matrix : np.ndarray    [n_rows, bow_vocab]  L2-normed BoW vectors
                                   aligned row-for-row with dedup_df
        """
        df = df.copy().reset_index(drop=True)

        # ── Stage A: exact fingerprint dedup ─────────────────────────────────
        df['exact_fp']  = df.apply(canonical_fingerprint,  axis=1)
        df['option_fp'] = df.apply(option_set_fingerprint, axis=1)
        n_before = len(df)
        df = (df.drop_duplicates(subset='exact_fp', keep='first')
                .reset_index(drop=True))
        logger.info(f"Exact dedup: {n_before} → {len(df)} "
                    f"(removed {n_before - len(df):,})")

        # ── Stage B: BoW encoding ─────────────────────────────────────────────
        # use ONLY the question text for semantic comparison
        q_texts = [normalize_text(str(r.get('prompt', '')))
                   for _, r in df.iterrows()]

        logger.info("Building BoW vectors …")
        bow_matrix = self.bow.fit_transform(q_texts)
        logger.info(f"BoW matrix: {bow_matrix.shape}")

        # ── Stage C: AgglomerativeClustering ──────────────────────────────────
        # distance_threshold = 1 - sim_threshold  (cosine distance)
        logger.info(f"Clustering (sim_threshold={self.sim_threshold}) …")
        clustering = AgglomerativeClustering(
            n_clusters         = None,
            distance_threshold = 1.0 - self.sim_threshold,
            metric             = 'cosine',
            linkage            = 'average',
        )
        labels               = clustering.fit_predict(bow_matrix)
        df['semantic_group'] = labels

        n_clusters = len(np.unique(labels))
        n_multi    = (pd.Series(labels).value_counts() > 1).sum()
        logger.info(f"Clusters: {n_clusters:,}  "
                    f"multi-member (near-dupes): {n_multi:,}")

        return df, bow_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Vocabulary  (for LSTM — separate from BoW vocab)
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD, UNK, SEP = '<PAD>', '<UNK>', '[SEP]'
    PAD_IDX = 0; UNK_IDX = 1; SEP_IDX = 2

    def __init__(self, max_vocab: int = 20_000, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.min_freq  = min_freq
        self.word2idx  : dict = {}
        self.idx2word  : dict = {}
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
        logger.info(f"LSTM vocab size: {len(self.word2idx):,}")
        return self

    def encode(self, text: str, max_len: int = None):
        toks = text.split()
        if max_len:
            toks = toks[:max_len]
        return [self.word2idx.get(t, self.UNK_IDX) for t in toks]

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MCQDataset(Dataset):

    def __init__(self, df, vocab, max_len: int = 180, is_test: bool = False):
        self.vocab   = vocab
        self.max_len = max_len
        self.is_test = is_test
        self.df      = df.reset_index(drop=True)
        self.data    = [self._process(i) for i in range(len(self.df))]

    def _process(self, i: int) -> dict:
        row = self.df.iloc[i]
        q   = normalize_text(str(row.get('prompt', '')))

        seqs, lens = [], []
        for lbl in ANSWER_LABELS:
            text = (f"{q} {self.vocab.SEP} "
                    f"{normalize_text(str(row.get(lbl, '')))}")
            ids  = self.vocab.encode(text, self.max_len)
            lens.append(max(len(ids), 1))
            ids += [self.vocab.PAD_IDX] * (self.max_len - len(ids))
            seqs.append(ids)

        item = dict(
            id      = row.get('id', i),
            options = torch.tensor(seqs, dtype=torch.long),
            lengths = torch.tensor(lens, dtype=torch.long),
        )
        if not self.is_test and 'answer' in row:
            lbl = str(row['answer']).strip().upper()
            item['label'] = torch.tensor(
                LABEL_TO_IDX.get(lbl, 0), dtype=torch.long)
        return item

    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch: list) -> dict:
    out = dict(
        id      = [b['id']      for b in batch],
        options = torch.stack([b['options'] for b in batch]),
        lengths = torch.stack([b['lengths'] for b in batch]),
    )
    if 'label' in batch[0]:
        out['label'] = torch.stack([b['label'] for b in batch])
    return out