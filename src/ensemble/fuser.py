# src/ensemble/fuser.py
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ScoreFuser:
    """
    Weighted average fusion of (n_samples, n_options) score matrices.

    Parameters
    ----------
    weights : optional {model_name: float}
        If None, equal weighting is applied.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights

    # ------------------------------------------------------------------
    @staticmethod
    def normalize(scores: np.ndarray) -> np.ndarray:
        """Per-sample min-max normalization → [0, 1]."""
        mins = scores.min(axis=1, keepdims=True)
        maxs = scores.max(axis=1, keepdims=True)
        denom = np.where((maxs - mins) == 0, 1.0, maxs - mins)
        return (scores - mins) / denom

    # ------------------------------------------------------------------
    def fuse(self, score_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        score_dict : {model_name: (n_samples, n_options)}

        Returns
        -------
        fused : (n_samples, n_options)
        """
        names = list(score_dict.keys())

        if self.weights is None:
            w_arr = np.ones(len(names)) / len(names)
        else:
            w_arr = np.array(
                [self.weights.get(n, 1.0) for n in names], dtype=float
            )
            w_arr /= w_arr.sum()

        fused = np.zeros_like(
            next(iter(score_dict.values())), dtype=float
        )
        for w, name in zip(w_arr, names):
            fused += w * self.normalize(score_dict[name])

        return fused