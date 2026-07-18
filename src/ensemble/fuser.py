# ensemble/fuser.py
# Score-level fusion with min-max normalisation + optional weight search.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.evaluator import MAPAtKEvaluator

logger = logging.getLogger("ScoreFuser")


class ScoreFuser:
    """
    Fuse score matrices from multiple rankers via weighted averaging.

    Score-level fusion is preferred over vote-level because it
    preserves confidence magnitudes across models.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights   # None → equal weighting

    # ─────────────────────────────────────────
    # Core fusion
    # ─────────────────────────────────────────

    @staticmethod
    def normalise(scores: np.ndarray) -> np.ndarray:
        """Row-wise min-max normalisation to [0, 1]."""
        lo    = scores.min(axis=1, keepdims=True)
        hi    = scores.max(axis=1, keepdims=True)
        denom = np.where(hi - lo > 0, hi - lo, 1.0)
        return (scores - lo) / denom

    def fuse(self, score_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine normalised score matrices.

        Parameters
        ----------
        score_dict : {model_name: (n_samples, n_options) array}

        Returns
        -------
        np.ndarray of shape (n_samples, n_options)
        """
        if not score_dict:
            raise ValueError("score_dict is empty.")

        normed = {k: self.normalise(v) for k, v in score_dict.items()}

        if self.weights is None:
            return np.mean(list(normed.values()), axis=0)

        total  = sum(self.weights.values())
        result = np.zeros_like(next(iter(normed.values())))
        for name, arr in normed.items():
            w = self.weights.get(name, 1.0) / total
            result += w * arr
        return result

    # ─────────────────────────────────────────
    # Grid-search for best weights
    # ─────────────────────────────────────────

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