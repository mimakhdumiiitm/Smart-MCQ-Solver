# models/tfidf_ranker.py
# TF-IDF cosine-similarity ranker with pickle-based persistence.
# On first run: fits the vectorizer and saves it.
# On subsequent runs: loads from disk — no re-training.

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config.config import Config
from utils.persistence import save_pickle, load_pickle

logger = logging.getLogger("TFIDFRanker")


class TFIDFRanker:
    """
    Rank MCQ options by TF-IDF cosine similarity with the question.

    Persistence
    -----------
    * `save()` — serialises the fitted vectorizer to a .pkl file.
    * `load()` — deserialises it; returns True on success.
    * `fit_or_load()` — convenience: tries to load first, fits if missing.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.vectorizer = TfidfVectorizer(
            max_features  = config.tfidf_max_features,
            ngram_range   = config.tfidf_ngram_range,
            min_df        = config.tfidf_min_df,
            sublinear_tf  = True,
            strip_accents = "unicode",
            analyzer      = "word",
            token_pattern = r"\w{1,}",
        )
        self._fitted = False

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Pickle the fitted TfidfVectorizer."""
        target = Path(path or self.cfg.tfidf_model_path)
        save_pickle(self.vectorizer, target)

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Attempt to load a saved vectorizer.

        Returns True if successful, False otherwise.
        """
        target = Path(path or self.cfg.tfidf_model_path)
        obj    = load_pickle(target)
        if obj is None:
            return False
        self.vectorizer = obj
        self._fitted    = True
        logger.info(
            f"TF-IDF vectorizer loaded "
            f"(vocab size: {len(self.vectorizer.vocabulary_)})"
        )
        return True

    # ─────────────────────────────────────────
    # Fit / fit-or-load
    # ─────────────────────────────────────────

    def fit(self, df) -> "TFIDFRanker":
        """
        Fit on the full corpus (prompts + all options).

        Parameters
        ----------
        df : preprocessed DataFrame with *_clean columns.
        """
        option_cols = [c for c in self.cfg.options if c in df.columns]
        corpus      = df["prompt_clean"].tolist()
        for opt in option_cols:
            corpus.extend(df[f"{opt}_clean"].tolist())
        corpus = [t for t in corpus if t.strip()]

        logger.info(f"Fitting TF-IDF on {len(corpus):,} texts …")
        self.vectorizer.fit(corpus)
        self._fitted = True
        logger.info(
            f"TF-IDF fitted (vocab size: {len(self.vectorizer.vocabulary_):,})"
        )
        self.save()   # auto-save after fitting
        return self

    def fit_or_load(self, df) -> "TFIDFRanker":
        """
        Load a cached model when `config.use_cached_models` is True
        and the file exists; otherwise fit from scratch and save.

        Usage
        -----
            ranker.fit_or_load(train_df)
        """
        if self.cfg.use_cached_models and self.load():
            return self
        return self.fit(df)

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────

    def predict_scores(self, df) -> np.ndarray:
        """
        Compute TF-IDF cosine similarity between every prompt and each option.

        Returns
        -------
        np.ndarray of shape (n_samples, n_options)
        """
        if not self._fitted:
            raise RuntimeError("Call fit_or_load() before predict_scores().")

        option_cols  = [c for c in self.cfg.options if c in df.columns]
        prompt_vecs  = self.vectorizer.transform(
            df["prompt_clean"].fillna("").tolist()
        )
        n_samples    = len(df)
        scores       = np.zeros((n_samples, len(option_cols)))

        for j, opt in enumerate(option_cols):
            opt_vecs  = self.vectorizer.transform(
                df[f"{opt}_clean"].fillna("").tolist()
            )
            dot       = np.array(prompt_vecs.multiply(opt_vecs).sum(axis=1)).flatten()
            p_norms   = np.array(prompt_vecs.power(2).sum(axis=1)).flatten() ** 0.5
            o_norms   = np.array(opt_vecs.power(2).sum(axis=1)).flatten()   ** 0.5
            denom     = np.where(p_norms * o_norms > 0, p_norms * o_norms, 1e-10)
            scores[:, j] = dot / denom

        return scores

    def predict_top_k(self, df, evaluator) -> List[List[str]]:
        """Return top-K ranked option labels for every sample."""
        option_cols = [c for c in self.cfg.options if c in df.columns]
        return evaluator.scores_to_top_k_predictions(
            self.predict_scores(df), option_cols
        )