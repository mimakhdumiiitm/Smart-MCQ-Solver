# src/ensemble/tta.py
from __future__ import annotations

import logging
from typing import Any, Callable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TestTimeAugmentor:
    """
    Test-Time Augmentation for MCQ via option-order permutation.

    Averaging over multiple option orderings reduces prediction variance,
    especially for borderline options.

    Parameters
    ----------
    config : Config
        Must expose ``config.options`` (list of option column names)
        and ``config.seed``.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _permute_options(
        self, df: pd.DataFrame, permutation: List[int]
    ) -> pd.DataFrame:
        """Return a copy of df with option columns reordered."""
        option_cols = [c for c in self.config.options if c in df.columns]
        original_cols = [f"{o}_clean" for o in option_cols]
        df_perm = df.copy()
        permuted_vals = [df[original_cols[i]].values for i in permutation]
        for j, col in enumerate(original_cols):
            df_perm[col] = permuted_vals[j]
        return df_perm

    # ------------------------------------------------------------------
    def augment_scores(
        self,
        score_fn: Callable[[pd.DataFrame], np.ndarray],
        df: pd.DataFrame,
        n_augmentations: int = 3,
    ) -> np.ndarray:
        """
        Average scores from original + (n_augmentations-1) permutations.

        Parameters
        ----------
        score_fn        : callable(df) → (n_samples, n_options) ndarray
        df              : input DataFrame
        n_augmentations : total number of augmented versions (incl. original)

        Returns
        -------
        averaged scores : (n_samples, n_options) ndarray
        """
        option_cols = [c for c in self.config.options if c in df.columns]
        n_options = len(option_cols)

        all_scores: list[np.ndarray] = [score_fn(df)]   # original

        rng = np.random.default_rng(self.config.seed)
        for _ in range(n_augmentations - 1):
            perm = rng.permutation(n_options).tolist()
            df_perm = self._permute_options(df, perm)
            perm_scores = score_fn(df_perm)
            inv_perm = np.argsort(perm)
            all_scores.append(perm_scores[:, inv_perm])

        return np.mean(all_scores, axis=0)