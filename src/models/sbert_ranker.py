# models/sbert_ranker.py
# Sentence-BERT ranker — no training needed, uses pre-trained weights.
# Embedding cache is optionally saved as a .pkl for speed.

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import Config
from utils.persistence import save_pickle, load_pickle

logger = logging.getLogger("SBERTRanker")


class SBERTRanker:
    """
    Rank options by cosine similarity of Sentence-BERT embeddings.

    The model weights are downloaded automatically by the
    sentence-transformers library — nothing to train.

    Embedding cache
    ---------------
    Encoding large DataFrames is expensive. Call `save_embedding_cache()`
    after the first run; subsequent runs call `load_embedding_cache()`.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        logger.info(f"Loading SBERT: {config.sbert_model}")
        self.model = SentenceTransformer(
            config.sbert_model, device=config.device
        )
        logger.info(
            f"SBERT ready — dim: "
            f"{self.model.get_sentence_embedding_dimension()}"
        )

    # ─────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────

    def encode(
        self,
        texts        : List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """L2-normalised embeddings; cosine sim = dot product."""
        return self.model.encode(
            texts,
            batch_size          = self.cfg.sbert_batch_size,
            show_progress_bar   = show_progress,
            normalize_embeddings= True,
            convert_to_numpy    = True,
        )

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────

    def predict_scores(self, df) -> np.ndarray:
        """
        Efficient batch encoding: encode ALL texts in a single pass.

        Returns
        -------
        np.ndarray of shape (n_samples, n_options)
        """
        option_cols    = [c for c in self.cfg.options if c in df.columns]
        n_opts         = len(option_cols)
        n_samples      = len(df)

        prompt_texts   = df["prompt_clean"].fillna("").tolist()
        flat_opt_texts = [
            text
            for opt in option_cols
            for text in df[f"{opt}_clean"].fillna("").tolist()
        ]

        logger.info(
            f"Encoding {n_samples} prompts + {len(flat_opt_texts)} options …"
        )
        prompt_embs = self.encode(prompt_texts, show_progress=True)
        opt_embs    = self.encode(flat_opt_texts, show_progress=True)

        # opt_embs is laid out [all_A, all_B, ...]; reshape to (n_opts, n, dim)
        opt_embs_3d = opt_embs.reshape(n_opts, n_samples, -1)
        # → (n_samples, n_opts, dim)
        opt_embs_3d = opt_embs_3d.transpose(1, 0, 2)

        # (n, dim) × (n, n_opts, dim) → (n, n_opts)
        scores = np.einsum("nd,nod->no", prompt_embs, opt_embs_3d)
        return scores

    def predict_top_k(self, df, evaluator) -> List[List[str]]:
        """Return top-K ranked option labels."""
        option_cols = [c for c in self.cfg.options if c in df.columns]
        return evaluator.scores_to_top_k_predictions(
            self.predict_scores(df), option_cols
        )

    # ─────────────────────────────────────────
    # Embedding cache
    # ─────────────────────────────────────────

    def build_embedding_cache(self, df) -> Dict[str, np.ndarray]:
        """
        Encode prompts and all options; return as a dict.

        Keys: "prompt", "A", "B", "C", "D", "E"  (whichever exist).
        """
        option_cols = [c for c in self.cfg.options if c in df.columns]
        cache: Dict[str, np.ndarray] = {}

        cache["prompt"] = self.encode(
            df["prompt_clean"].fillna("").tolist(), show_progress=True
        )
        for opt in option_cols:
            cache[opt] = self.encode(
                df[f"{opt}_clean"].fillna("").tolist(), show_progress=True
            )
        return cache

    def save_embedding_cache(
        self,
        cache   : Dict[str, np.ndarray],
        filename: str,
    ) -> None:
        """Persist an embedding dict to disk."""
        path = Path(self.cfg.model_dir) / filename
        save_pickle(cache, path)

    def load_embedding_cache(
        self,
        filename: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load a previously saved embedding dict."""
        path = Path(self.cfg.model_dir) / filename
        return load_pickle(path)


"""
-- understand contexual meaning 
-- build on top of BERT 
-- gives embedding for each sentence
-- internnal working - each word has embedding, then it is passed through transformer layers to get contextual embedding for each word, then pooling is done to get sentence embedding
-- cosine similarity is used to compare embeddings of prompt and options
-- Normalization is done to get cosine similarity as dot product
"""