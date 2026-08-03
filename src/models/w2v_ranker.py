# models/w2v_ranker.py
# Word2Vec mean-pooling ranker with Gensim native-format persistence.

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from config.config import Config
from utils.persistence import save_w2v, load_w2v

logger = logging.getLogger("Word2VecRanker")


class Word2VecRanker:
    """
    Rank MCQ options by cosine similarity of mean-pooled Word2Vec vectors.

    Persistence
    -----------
    Uses Gensim's native .model format (not pickle) for full compatibility
    with Gensim's KeyedVectors interface.
    """

    def __init__(self, config: Config) -> None:
        self.cfg   = config
        self.model: Optional[Word2Vec] = None

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Save the Gensim Word2Vec model."""
        if self.model is None:
            raise RuntimeError("No model to save — call fit() first.")
        save_w2v(self.model, path or self.cfg.w2v_model_path)

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load a saved Gensim Word2Vec model.

        Returns True if successful.
        """
        obj = load_w2v(path or self.cfg.w2v_model_path)
        if obj is None:
            return False
        self.model = obj
        logger.info(
            f"Word2Vec loaded (vocab: {len(self.model.wv):,} words)"
        )
        return True

    # ─────────────────────────────────────────
    # Fit / fit-or-load
    # ─────────────────────────────────────────

    def fit(self, *dfs) -> "Word2VecRanker":
        """
        Train Word2Vec on one or more DataFrames (e.g. train + test).

        Including test text improves vocabulary coverage without leaking labels.

        Parameters
        ----------
        *dfs : one or more preprocessed DataFrames.
        """
        option_cols = [c for c in self.cfg.options if c in dfs[0].columns]
        sentences   : List[List[str]] = []

        for df in dfs:
            for _, row in df.iterrows():
                sentences.append(self._tok(str(row.get("prompt_clean", ""))))
                for opt in option_cols:
                    sentences.append(self._tok(str(row.get(f"{opt}_clean", ""))))

        sentences = [s for s in sentences if s]
        logger.info(f"Training Word2Vec on {len(sentences):,} sentences …")

        self.model = Word2Vec(
            sentences   = sentences,
            vector_size = self.cfg.w2v_vector_size,
            window      = self.cfg.w2v_window,
            min_count   = self.cfg.w2v_min_count,
            epochs      = self.cfg.w2v_epochs,
            workers     = min(4, os.cpu_count() or 1),
            sg          = 1,   # skip-gram
            hs          = 0,   # negative sampling
            negative    = 5,
        )
        logger.info(
            f"Word2Vec trained (vocab: {len(self.model.wv):,} words)"
        )
        self.save()   # auto-save
        return self

    def fit_or_load(self, *dfs) -> "Word2VecRanker":
        """Load cached model if available; otherwise train and save."""
        if self.cfg.use_cached_models and self.load():
            return self
        return self.fit(*dfs)

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────

    def predict_scores(self, df) -> np.ndarray:
        """
        Cosine similarities between prompt and each option in Word2Vec space.

        Returns
        -------
        np.ndarray of shape (n_samples, n_options)
        """
        if self.model is None:
            raise RuntimeError("Call fit_or_load() before predict_scores().")

        option_cols   = [c for c in self.cfg.options if c in df.columns]
        prompt_vecs   = np.array([
            self._embed(t) for t in df["prompt_clean"].fillna("")
        ])
        scores        = np.zeros((len(df), len(option_cols)))

        for j, opt in enumerate(option_cols):
            opt_vecs     = np.array([
                self._embed(t) for t in df[f"{opt}_clean"].fillna("")
            ])
            dots         = np.einsum("ij,ij->i", prompt_vecs, opt_vecs)
            p_norms      = np.linalg.norm(prompt_vecs, axis=1)
            o_norms      = np.linalg.norm(opt_vecs,    axis=1)
            denom        = np.where(p_norms * o_norms > 0, p_norms * o_norms, 1e-10)
            scores[:, j] = dots / denom

        return scores

    def predict_top_k(self, df, evaluator) -> List[List[str]]:
        """Return top-K ranked option labels."""
        option_cols = [c for c in self.cfg.options if c in df.columns]
        return evaluator.scores_to_top_k_predictions(
            self.predict_scores(df), option_cols
        )

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _tok(text: str) -> List[str]:
        return simple_preprocess(text, deacc=True, min_len=2)

    def _embed(self, text: str) -> np.ndarray:
        """Mean-pool Word2Vec vectors; return zero vector if OOV."""
        vecs = [
            self.model.wv[t]
            for t in self._tok(str(text))
            if t in self.model.wv
        ]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.cfg.w2v_vector_size)


"""
neural network-based technique ---> converts words into vectors
-- Each Word converted to a vector based  on nearby words 
--- 1. CBOW (Chain of Bag of words) :  predicts missing word
--- 2. Skip Gram : predicts nearby words based on a given word
-- word vector --> senetence vector : mean pooling
-- cosine similarity between sentence vectors of prompt and options

problems :
x Understands meaning
x Word order
x Sentence embedding
x Pretrained knowledge
"""