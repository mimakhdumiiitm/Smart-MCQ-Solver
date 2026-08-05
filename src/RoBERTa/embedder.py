# src/RoBERTa/embedder.py
"""
SBERT embedder — identical interface to DeBERTa version.
Reused for dedup and leakage audit.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger("RoBERTa.Embedder")


class SBERTEmbedder:
    """
    Thin wrapper around sentence-transformers.

    Parameters
    ──────────
    model_name  : HuggingFace model ID or local path
    batch_size  : sentences per forward pass
    device      : "cuda" | "cpu" | None (auto)
    normalize   : L2-normalise so dot == cosine  (default True)
    """

    def __init__(
        self,
        model_name : str  = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size : int  = 256,
        device     : str  = None,
        normalize  : bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.\n"
                "pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize

        logger.info(f"Loading SBERT: {model_name} …")
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("SBERT ready.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Returns float32 matrix [N, 384], L2-normalised if normalize=True.
        """
        logger.info(f"Encoding {len(texts):,} texts with SBERT …")
        vecs = self._model.encode(
            texts,
            batch_size           = self.batch_size,
            show_progress_bar    = True,
            convert_to_numpy     = True,
            normalize_embeddings = self.normalize,
        )
        logger.info(f"SBERT embeddings: {vecs.shape}  dtype={vecs.dtype}")
        return vecs.astype(np.float32)

    def __repr__(self):
        return f"SBERTEmbedder(model='{self.model_name}')"