# src/ensemble/fuser.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.evaluation.evaluator import MAPAtKEvaluator

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

     # ─────────────────────────────────────────
    # Grid-search for best weights
    @staticmethod
    def grid_search(
        score_dict  : Dict[str, np.ndarray],
        actuals     : List[str],
        evaluator   : MAPAtKEvaluator,
        option_cols : List[str],
        weight_grid : Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Exhaustive grid search over per-model weights.

        Parameters
        ----------
        score_dict  : scores from each model (aligned rows).
        actuals     : ground-truth answer labels.
        evaluator   : MAPAtKEvaluator instance.
        option_cols : list of option labels, e.g. ["A","B","C","D","E"].
        weight_grid : {model_name: [candidate_weights]}.
                      Defaults to {0.0, 0.5, 1.0, 2.0} for non-SBERT,
                      {1.0, 2.0, 3.0} for SBERT.

        Returns
        -------
        (best_weights_dict, best_map_score)
        """
        if weight_grid is None:
            weight_grid = {
                name: ([1.0, 2.0, 3.0] if "sbert" in name.lower()
                       else [0.0, 0.5, 1.0, 2.0])
                for name in score_dict
            }

        import itertools
        model_names   = list(score_dict.keys())
        grid_values   = [weight_grid[n] for n in model_names]

        best_map      = -1.0
        best_weights  : Dict[str, float] = {}

        for combo in itertools.product(*grid_values):
            weights  = dict(zip(model_names, combo))
            fuser    = ScoreFuser(weights=weights)
            fused    = fuser.fuse(score_dict)
            preds    = evaluator.scores_to_top_k_predictions(fused, option_cols)
            map_val  = evaluator.mean_average_precision_at_k(actuals, preds)

            if map_val > best_map:
                best_map     = map_val
                best_weights = weights

        logger.info(f"Grid search complete — best weights: {best_weights}")
        logger.info(f"Best MAP@{evaluator.k}: {best_map:.4f}")
        return best_weights, best_map