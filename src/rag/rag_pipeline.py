"""
rag_pipeline.py
===============
Milestone 3: Retrieval-Augmented Generation (RAG) pipeline.

Contains:
- RAGPipeline  : FAISS-backed retrieval + answer-vote and semantic scoring

Reuse policy:
- Checks for saved .npy score files before recomputing
- Reuses processed CSVs already on disk (train_processed.csv / test_processed.csv)
- Falls back to CPU FAISS when GPU FAISS is unavailable
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_none(path: Path) -> Optional[np.ndarray]:
    """Return cached numpy array or None."""
    if path.exists():
        logger.info(f"[cache] Reusing {path.name}")
        return np.load(path)
    return None


def _both_exist(p1: Path, p2: Path) -> bool:
    return p1.exists() and p2.exists()


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieval-Augmented MCQ scoring pipeline.

    Steps
    -----
    1. Encode all training prompts with Sentence-BERT.
    2. Build a FAISS index (GPU if available, else CPU).
    3. For each query, retrieve the K most similar training examples.
    4. Produce two complementary score matrices:
       - vote scores    : weighted answer-label votes from retrieved rows
       - semantic scores: cosine similarity of current options to the
                          mean embedding of retrieved correct-answer texts

    Parameters
    ----------
    config               : project Config object
    retrieval_model_name : any sentence-transformers model identifier
    top_k_retrieve       : number of neighbours to retrieve
    """

    def __init__(
        self,
        config,
        retrieval_model_name: str = "all-mpnet-base-v2",
        top_k_retrieve:       int = 5,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.config          = config
        self.top_k_retrieve  = top_k_retrieve
        self.logger          = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"Loading retrieval model: {retrieval_model_name}")
        self.retrieval_model = SentenceTransformer(
            retrieval_model_name,
            device=config.device,
        )

        # State populated by build_index()
        self.index:            Optional[object]      = None   # faiss.Index
        self.train_df:         Optional[pd.DataFrame] = None
        self.train_embeddings: Optional[np.ndarray]  = None

    # ── index construction ────────────────────────────────────────────────────

    def build_index(self, train_df: pd.DataFrame) -> "RAGPipeline":
        """
        Encode training prompts and build a FAISS IndexFlatIP.

        Uses L2-normalised vectors so inner product == cosine similarity.
        GPU FAISS is attempted first; silently falls back to CPU.

        Parameters
        ----------
        train_df : full training DataFrame (with ``prompt_clean`` column)
        """
        import faiss

        self.train_df = train_df.reset_index(drop=True)
        prompts = train_df["prompt_clean"].fillna("").tolist()

        self.logger.info(f"Encoding {len(prompts)} training prompts …")
        self.train_embeddings = self.retrieval_model.encode(
            prompts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = self.train_embeddings.shape[1]
        self.logger.info(f"Embedding dim={dim}")

        # ── FAISS index (GPU preferred) ───────────────────────────────────
        try:
            res       = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            self.logger.info("FAISS: using GPU index.")
        except Exception:
            self.index = faiss.IndexFlatIP(dim)
            self.logger.info("FAISS: using CPU index (GPU unavailable).")

        self.index.add(self.train_embeddings)
        self.logger.info(f"FAISS index ready  ({self.index.ntotal} vectors).")

        return self   # allow chaining

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the top-K nearest neighbours for each query.

        Returns
        -------
        distances : (n_queries, top_k)  cosine similarities ∈ [-1, 1]
        indices   : (n_queries, top_k)  row indices into self.train_df
        """
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieve().")

        query_embs = self.retrieval_model.encode(
            query_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        return self.index.search(query_embs, self.top_k_retrieve)

    # ── scoring methods ───────────────────────────────────────────────────────

    def compute_rag_scores(
        self,
        query_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Answer-label vote scores.

        For each query:
        - Retrieve top-K training rows.
        - Accumulate cosine-similarity-weighted votes for each answer label.

        Returns
        -------
        rag_scores : (n_samples, n_options)
        """
        option_cols = [c for c in self.config.options if c in query_df.columns]
        n_samples   = len(query_df)
        n_options   = len(option_cols)

        query_texts        = query_df["prompt_clean"].fillna("").tolist()
        distances, indices = self.retrieve(query_texts)

        rag_scores = np.zeros((n_samples, n_options))

        if self.config.answer_col not in self.train_df.columns:
            self.logger.warning(
                "No answer column in train_df – returning zero RAG vote scores."
            )
            return rag_scores

        for i in range(n_samples):
            vote_weights = {opt: 0.0 for opt in option_cols}

            for idx, dist in zip(indices[i], distances[i]):
                if idx < 0 or idx >= len(self.train_df):
                    continue
                answer = self.train_df.iloc[idx].get(self.config.answer_col)
                if answer in vote_weights:
                    vote_weights[answer] += float(dist)

            for j, opt in enumerate(option_cols):
                rag_scores[i, j] = vote_weights[opt]

        return rag_scores

    def compute_semantic_context_scores(
        self,
        query_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Semantic similarity of current options to retrieved correct-answer texts.

        For each query:
        1. Retrieve top-K training rows.
        2. Collect the correct-answer text from each retrieved row.
        3. Compute the mean embedding of those correct answers.
        4. Score each current option by cosine similarity to that mean.

        Returns
        -------
        semantic_scores : (n_samples, n_options)
        """
        option_cols = [c for c in self.config.options if c in query_df.columns]
        n_samples   = len(query_df)
        n_options   = len(option_cols)

        query_texts        = query_df["prompt_clean"].fillna("").tolist()
        _, indices         = self.retrieve(query_texts)

        semantic_scores = np.zeros((n_samples, n_options))

        if self.config.answer_col not in self.train_df.columns:
            return semantic_scores

        for i in range(n_samples):
            valid_idx = [
                idx for idx in indices[i]
                if 0 <= idx < len(self.train_df)
            ]

            # ── collect correct-answer texts from retrieved rows ───────────
            correct_texts: List[str] = []
            for idx in valid_idx:
                row    = self.train_df.iloc[idx]
                label  = row.get(self.config.answer_col)
                if label and f"{label}_clean" in self.train_df.columns:
                    text = row.get(f"{label}_clean", "")
                    if text:
                        correct_texts.append(text)

            if not correct_texts:
                continue

            # Mean embedding of retrieved correct answers
            correct_embs = self.retrieval_model.encode(
                correct_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            mean_emb = correct_embs.mean(axis=0)   # (D,)

            # ── score each option ─────────────────────────────────────────
            for j, opt in enumerate(option_cols):
                opt_text = query_df.iloc[i].get(f"{opt}_clean", "")
                if not opt_text:
                    continue

                opt_emb = self.retrieval_model.encode(
                    [opt_text],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)[0]

                semantic_scores[i, j] = float(np.dot(opt_emb, mean_emb))

        return semantic_scores

    # ── context text for fine-tuning prompt augmentation ──────────────────────

    def get_retrieval_context(
        self,
        query_df: pd.DataFrame,
        top_k:    int = 3,
    ) -> List[str]:
        """
        Build a short text context string per query from the top-K retrieved rows.

        Format: "Q: <question> A: <correct answer text> | Q: … A: …"

        Useful as prompt prefix for fine-tuned generative models.

        Parameters
        ----------
        query_df : DataFrame with ``prompt_clean``
        top_k    : how many retrieved rows to include in the context string
        """
        option_cols = [c for c in self.config.options if c in query_df.columns]
        query_texts = query_df["prompt_clean"].fillna("").tolist()
        _, indices  = self.retrieve(query_texts)

        contexts: List[str] = []

        for i in range(len(query_df)):
            valid_idx = [
                idx for idx in indices[i]
                if 0 <= idx < len(self.train_df)
            ][:top_k]

            parts: List[str] = []
            for idx in valid_idx:
                row         = self.train_df.iloc[idx]
                question    = row.get("prompt", "")
                answer_lbl  = row.get(self.config.answer_col, "")
                answer_text = row.get(answer_lbl, "") if answer_lbl else ""
                if question and answer_text:
                    parts.append(f"Q: {question} A: {answer_text}")

            contexts.append(" | ".join(parts))

        return contexts


"""
             Question
                 │
                 ▼
        Convert into vector (sentence BERT)
                 │
                 ▼
      Search similar vectors (FAISS indexing)
          (using FAISS)
                 │
                 ▼
     Retrieve similar questions
                 │
                 ▼
    Use retrieved information
                 │
                 ▼
     Predict final answer

2 types of scoring

1. rag score - we have qeusions with their answers and similariy score 
Q1 → Answer = A → Similarity = 0.95
Q2 → Answer = A → Similarity = 0.92
Q3 → Answer = A → Similarity = 0.90
Option A
0.95 + 0.92 + 0.90
= 2.77

2. semantic score - we have questions with their answers and similariy score
mean vector of all the answers of the retrieved questions and now check similarity of the mean vector with the options of the current question
"""