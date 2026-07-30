# src/ensemble/rank_averager.py
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class RankAverager:
    """
    Rank-based ensemble averaging.

    Converts scores → ordinal ranks per sample, then averages across
    models.  Robust to scale differences and outlier confidence values.

    Higher average rank = better option (consistent with score convention).
    """

    @staticmethod
    def average_ranks(score_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        score_dict : {model_name: (n_samples, n_options)}

        Returns
        -------
        avg_rank_scores : (n_samples, n_options)  – higher is better
        """
        all_ranks = []
        for scores in score_dict.values():
            # best option → rank n_options, worst → rank 1
            ranks = np.argsort(np.argsort(scores, axis=1), axis=1) + 1
            all_ranks.append(ranks)
        return np.mean(all_ranks, axis=0)