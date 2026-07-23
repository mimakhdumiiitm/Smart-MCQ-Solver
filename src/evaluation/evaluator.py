# evaluation/evaluator.py
# MAP@K metric — identical to the Kaggle evaluation script.
# Kept in its own module so every phase imports the SAME implementation.

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    )
logger = logging.getLogger("Evaluator")

# W&B is optional; import lazily to avoid hard dependency
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class MAPAtKEvaluator:
    """
    Mean Average Precision @ K.

    For single-answer MCQ:
        AP@K = 1 / rank   if correct answer is in top-K predictions
               0           otherwise

    MAP@K = mean of AP@K over all samples.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k

    # ─────────────────────────────────────────
    # Core metric
    # ─────────────────────────────────────────

    def average_precision_at_k(
        self, actual: str, predicted: List[str]
    ) -> float:
        """AP@K for one sample."""
        for rank, pred in enumerate(predicted[: self.k], start=1):
            if pred == actual:
                return 1.0 / rank
        return 0.0

    def mean_average_precision_at_k(
        self,
        actuals    : List[str],
        predictions: List[List[str]],
    ) -> float:
        """MAP@K across all samples."""
        if len(actuals) != len(predictions):
            raise ValueError(
                f"Length mismatch: actuals={len(actuals)}, "
                f"predictions={len(predictions)}"
            )
        scores = [
            self.average_precision_at_k(a, p)
            for a, p in zip(actuals, predictions)
        ]
        return float(np.mean(scores))

    # ─────────────────────────────────────────
    # Full evaluation with breakdown
    # ─────────────────────────────────────────

    def evaluate(
        self,
        df         : pd.DataFrame,
        predictions: List[List[str]],
        answer_col : str = "answer",
        split      : str = "val",
        wandb_run  : Optional[object] = None,
    ) -> Dict[str, float]:
        """
        Compute MAP@K and per-rank hit rates.

        Parameters
        ----------
        df          : DataFrame containing the ground-truth answer column.
        predictions : Top-K prediction lists aligned with df rows.
        answer_col  : Name of the ground-truth column.
        split       : Label prefix for logged metrics (e.g. "tfidf_val").
        wandb_run   : Active wandb run (or None to skip logging).

        Returns
        -------
        dict of metric_name → float
        """
        if answer_col not in df.columns:
            raise ValueError(f"Column '{answer_col}' not found in DataFrame.")

        actuals   = df[answer_col].tolist()
        map_score = self.mean_average_precision_at_k(actuals, predictions)

        pos_counts: Dict = {1: 0, 2: 0, 3: 0, "missed": 0}
        for actual, pred in zip(actuals, predictions):
            top = pred[: self.k]
            if actual in top:
                pos_counts[top.index(actual) + 1] += 1
            else:
                pos_counts["missed"] += 1

        n = len(actuals)
        metrics = {
            f"{split}/map@{self.k}"  : map_score,
            f"{split}/hit@1"         : pos_counts[1]        / n,
            f"{split}/hit@2"         : pos_counts[2]        / n,
            f"{split}/hit@3"         : pos_counts[3]        / n,
            f"{split}/missed_rate"   : pos_counts["missed"] / n,
        }

        logger.info(f"\n{'─'*40}\nEvaluation [{split.upper()}]\n{'─'*40}")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")
        logger.info(f"  Positions: {pos_counts}")

        if wandb_run is not None and _WANDB_AVAILABLE:
            wandb_run.log(metrics)

        return metrics

    # ─────────────────────────────────────────
    # Score → ranked predictions
    # ─────────────────────────────────────────

    def scores_to_top_k_predictions(
        self,
        scores       : np.ndarray,
        option_labels: List[str],
    ) -> List[List[str]]:
        """
        Convert a (n_samples, n_options) score matrix to ranked label lists.

        Higher score = preferred option.
        """
        predictions = []
        for row in scores:
            ranked = np.argsort(-row)
            predictions.append(
                [option_labels[i] for i in ranked[: self.k]]
            )
        return predictions

    # ─────────────────────────────────────────
    # Unit-test helper
    # ─────────────────────────────────────────

    @staticmethod
    def run_unit_tests() -> None:
        """Smoke-test the MAP@K implementation."""
        ev = MAPAtKEvaluator(k=3)
        assert ev.average_precision_at_k("A", ["A", "B", "C"]) == 1.0
        assert ev.average_precision_at_k("B", ["A", "B", "C"]) == 0.5
        assert abs(ev.average_precision_at_k("C", ["A", "B", "C"]) - 1/3) < 1e-9
        assert ev.average_precision_at_k("D", ["A", "B", "C"]) == 0.0
        assert ev.average_precision_at_k("A", [])              == 0.0

        map_val = ev.mean_average_precision_at_k(
            actuals     = ["A", "B", "C"],
            predictions = [["A","B","C"], ["A","B","C"], ["A","B","C"]],
        )
        expected = (1.0 + 0.5 + 1/3) / 3
        assert abs(map_val - expected) < 1e-9

        logger.info("All MAP@3 unit tests passed ✓")

     # updated 
    def compute_classification_metrics(
        self,
        actuals   : List[str],
        predictions: List[List[str]],
    ) -> Dict[str, float]:

        # Use only the top-1 prediction for classification metrics
        top1_preds = [p[0] if p else "" for p in predictions]

        return {
            "f1_score" : f1_score(
                actuals, top1_preds, average="macro", zero_division=0
            ),
            "accuracy" : accuracy_score(actuals, top1_preds),
            "precision": precision_score(
                actuals, top1_preds, average="macro", zero_division=0
            ),
            "recall"   : recall_score(
                actuals, top1_preds, average="macro", zero_division=0
            ),
        }