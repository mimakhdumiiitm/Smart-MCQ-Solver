# src/DeBERTa/embedder.py
"""
SBERT-based semantic embedder.

Replaces the hand-crafted BoW encoder.
Uses sentence-transformers/all-MiniLM-L6-v2:
  - 384-dim embeddings
  - Very fast (CPU or GPU)
  - L2-normalised output  →  dot product == cosine similarity
  - Pretrained on 1B+ sentence pairs

Used for:
  1. Semantic deduplication  (AgglomerativeClustering on cosine distance)
  2. Leakage audit           (cosine similarity matrix train × val)
  3. Sim-distribution plot

NOT used for MCQ classification (DeBERTa handles that).
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger("DeBERTa.Embedder")


class SBERTEmbedder:
    """
    Thin wrapper around sentence-transformers.

    Parameters
    ──────────
    model_name  : HuggingFace model ID or local path
    batch_size  : sentences per forward pass
    device      : "cuda" | "cpu" | None (auto)
    normalize   : if True, L2-normalise so dot == cosine  (default True)
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
                "Install: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize

        logger.info(f"Loading SBERT: {model_name} …")
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("SBERT ready.")

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings.

        Returns
        ───────
        float32 matrix [N, 384], L2-normalised if normalize=True.
        """
        logger.info(f"Encoding {len(texts):,} texts with SBERT …")
        vecs = self._model.encode(
            texts,
            batch_size          = self.batch_size,
            show_progress_bar   = True,
            convert_to_numpy    = True,
            normalize_embeddings= self.normalize,
        )
        logger.info(f"SBERT embeddings: {vecs.shape}  "
                    f"dtype={vecs.dtype}")
        return vecs.astype(np.float32)

    def __repr__(self):
        return f"SBERTEmbedder(model='{self.model_name}')"