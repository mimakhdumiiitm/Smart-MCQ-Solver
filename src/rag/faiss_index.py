# src/rag/faiss_index.py
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Thin wrapper around a flat L2 FAISS index.

    Encapsulates build / search so RAGPipeline stays clean.
    Falls back to brute-force numpy cosine search when faiss is not
    available (e.g. CPU-only Kaggle kernel without faiss-cpu installed).
    """

    def __init__(self, dim: int, use_gpu: bool = False) -> None:
        self.dim = dim
        self.use_gpu = use_gpu
        self._index = None
        self._fallback = False
        self._vectors: np.ndarray | None = None  # kept for fallback

    # ------------------------------------------------------------------
    def build(self, vectors: np.ndarray) -> None:
        """
        Index the given (n, dim) float32 embedding matrix.

        Parameters
        ----------
        vectors : (n_docs, dim) float32
        """
        vectors = vectors.astype(np.float32)
        self._vectors = vectors

        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatL2(self.dim)
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("FAISS index on GPU.")
                except Exception:
                    logger.warning("GPU FAISS failed – using CPU.")
            index.add(vectors)
            self._index = index
            logger.info(
                f"FAISS index built: {index.ntotal} vectors, dim={self.dim}"
            )
        except ImportError:
            logger.warning(
                "faiss not available – using numpy brute-force search."
            )
            self._fallback = True

    # ------------------------------------------------------------------
    def search(
        self, query_vectors: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top_k nearest neighbours for each query.

        Returns
        -------
        distances : (n_queries, top_k)
        indices   : (n_queries, top_k)
        """
        query_vectors = query_vectors.astype(np.float32)

        if self._fallback:
            # Brute-force cosine similarity (negated for consistency)
            norms_q = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            norms_d = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            sim = (query_vectors / (norms_q + 1e-8)) @ \
                  (self._vectors / (norms_d + 1e-8)).T  # (n_q, n_d)
            indices = np.argsort(-sim, axis=1)[:, :top_k]
            distances = -np.take_along_axis(sim, indices, axis=1)
            return distances, indices

        distances, indices = self._index.search(query_vectors, top_k)
        return distances, indices