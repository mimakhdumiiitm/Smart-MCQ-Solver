# models/tfidf_ranker.py
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config.config import Config
from utils.persistence import save_pickle, load_pickle

logger = logging.getLogger("TFIDFRanker")


class TFIDFRanker:
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

"""
information retrieval technique ---> ranks documents (or text) based on how similar they are to a query.
step 1 - Preprocessing 
Step 2 -  Build Vovabulary
Step 3 -  Calculate Term Frquencey - TF = Count of word in document / Total number of words in document
Step 4 - Calculate Inverse Document Frequency - IDF = log(Total number of documents / Number of documents containing the word)
Step 5 - Calculate TF-IDF = TF * IDF
Step 6 - Calculate Cosine Similarity between the prompt and each option
Step 7 - Rank the answer options based on their cosine similarity scores.

problems :
x Understands meaning
x Word order
x Sentence embedding
x Pretrained knowledge
"""