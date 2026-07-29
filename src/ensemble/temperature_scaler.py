# src/ensemble/temperature_scaler.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from src.evaluation.evaluator import MAPAtKEvaluator

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """
    Scalar temperature calibration.

    T > 1  → softer (less confident)
    T < 1  → sharper (more confident)

    Optimal T found via bounded scalar minimisation on the val set.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0

    # ------------------------------------------------------------------
    def fit(
        self,
        logits: np.ndarray,
        labels: List[str],
        option_cols: List[str],
        evaluator: "MAPAtKEvaluator",
    ) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar
        from scipy.special import softmax as sp_softmax

        def neg_map(T: float) -> float:
            probs = sp_softmax(logits / T, axis=1)
            preds = evaluator.scores_to_top_k_predictions(probs, option_cols)
            return -evaluator.mean_average_precision_at_k(labels, preds)

        result = minimize_scalar(
            neg_map, bounds=(0.1, 5.0), method="bounded",
            options={"xatol": 1e-4}
        )
        self.temperature = float(result.x)
        logger.info(
            f"Optimal T={self.temperature:.4f}  "
            f"(MAP@3={-result.fun:.4f})"
        )
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def fit_transform(
        self,
        logits: np.ndarray,
        labels: List[str],
        option_cols: List[str],
        evaluator: "MAPAtKEvaluator",
    ) -> np.ndarray:
        self.fit(logits, labels, option_cols, evaluator)
        return self.transform(logits)