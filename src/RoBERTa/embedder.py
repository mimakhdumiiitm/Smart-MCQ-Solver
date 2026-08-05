# src/RoBERTa/embedder.py
"""
SBERT embedder — optimised for throughput.

Changes from reviewed version
──────────────────────────────
  - encode() accepts an optional cache_path; returns cached embeddings
    if the file exists, saving re-encoding on repeated runs.
  - progress bar suppressed when logging level > INFO.
  - dtype cast moved here (was duplicated in callers).
  - __repr__ improved.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("RoBERTa.Embedder")


class SBERTEmbedder:
    """
    Thin wrapper around sentence-transformers.

    Parameters
    ──────────
    model_name  : HuggingFace model ID or local path
    batch_size  : sentences per forward pass
    device      : "cuda" | "cpu" | None (auto-detect)
    normalize   : L2-normalise outputs so dot-product == cosine similarity
    """

    def __init__(
        self,
        model_name : str   = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size : int   = 256,
        device     : Optional[str] = None,
        normalize  : bool  = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize

        logger.info("Loading SBERT: %s …", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("SBERT ready.")

    # ──────────────────────────────────────────────────────────────────────

    def encode(
        self,
        texts      : List[str],
        cache_path : Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode *texts* and return float32 matrix [N, dim].

        If *cache_path* is given and the file exists the cached array is
        returned immediately (no GPU work).  If it does not exist the
        embeddings are computed and saved there for future calls.
        """
        if cache_path is not None:
            p = Path(cache_path)
            if p.exists():
                arr = np.load(p).astype(np.float32)
                logger.info(
                    "SBERT cache hit: %s  shape=%s", p.name, arr.shape
                )
                return arr

        show_bar = logger.isEnabledFor(logging.INFO)
        logger.info("Encoding %d texts with SBERT …", len(texts))

        vecs: np.ndarray = self._model.encode(
            texts,
            batch_size           = self.batch_size,
            show_progress_bar    = show_bar,
            convert_to_numpy     = True,
            normalize_embeddings = self.normalize,
        )
        vecs = vecs.astype(np.float32)
        logger.info("SBERT embeddings: %s  dtype=%s", vecs.shape, vecs.dtype)

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vecs)
            logger.info("SBERT embeddings cached → %s", cache_path)

        return vecs

    def __repr__(self) -> str:
        return (
            f"SBERTEmbedder(model='{self.model_name}', "
            f"batch_size={self.batch_size}, normalize={self.normalize})"
        )