# src/ensemble/orchestrator.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ensemble.fuser import ScoreFuser
from src.ensemble.rank_averager import RankAverager
from src.ensemble.temperature_scaler import TemperatureScaler

logger = logging.getLogger(__name__)


class EnsembleOrchestrator:
    """
    Ties together all ensemble strategies and auto-selects the best.

    Strategies
    ----------
    weighted_score  – calibrate each model, optimise weights on val MAP@3
    rank_average    – model-agnostic rank fusion (scale-invariant)
    soft_vote       – softmax probabilities averaged uniformly

    Parameters
    ----------
    config      : Config  (needs .options, .seed)
    evaluator   : MAPAtKEvaluator  (needs .scores_to_top_k_predictions,
                                          .mean_average_precision_at_k)
    option_cols : list[str]  option column names e.g. ["A","B","C","D","E"]
    """

    def __init__(
        self,
        config: Any,
        evaluator: Any,
        option_cols: List[str],
    ) -> None:
        self.config      = config
        self.evaluator   = evaluator
        self.option_cols = option_cols
        self.rank_avg    = RankAverager()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def _map3(self, scores: np.ndarray, labels: List[str]) -> float:
        preds = self.evaluator.scores_to_top_k_predictions(
            scores, self.option_cols
        )
        return self.evaluator.mean_average_precision_at_k(labels, preds)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def calibrate_all(
        self,
        score_dict: Dict[str, np.ndarray],
        labels: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Fit one TemperatureScaler per model on the validation set.

        Returns
        -------
        calibrated_scores : dict
        temperatures      : {model_name: T}  – reuse on test set
        """
        calibrated: Dict[str, np.ndarray] = {}
        temperatures: Dict[str, float]    = {}

        for name, scores in score_dict.items():
            ts = TemperatureScaler()
            calibrated[name]    = ts.fit_transform(
                scores, labels, self.option_cols, self.evaluator
            )
            temperatures[name]  = ts.temperature
            logger.info(f"[calibrate] {name}  T={ts.temperature:.4f}")

            try:
                import wandb
                if wandb.run:
                    wandb.log({f"calibration/{name}_temperature": ts.temperature})
            except Exception:
                pass

        return calibrated, temperatures

    # ------------------------------------------------------------------
    # weight optimisation
    # ------------------------------------------------------------------

    def find_optimal_weights(
        self,
        val_score_dict: Dict[str, np.ndarray],
        labels: List[str],
    ) -> Dict[str, float]:
        """
        Nelder-Mead optimisation of ensemble weights on val MAP@3.
        Initialised proportional to each model's individual MAP@3.
        """
        from scipy.optimize import minimize

        names    = list(val_score_dict.keys())
        n_models = len(names)

        # individual baselines
        init_maps = []
        for name, scores in val_score_dict.items():
            m = self._map3(scores, labels)
            init_maps.append(m)
            logger.info(f"  {name}  base MAP@3 = {m:.4f}")

        total        = sum(init_maps)
        init_weights = (
            np.array(init_maps) / total
            if total > 0
            else np.ones(n_models) / n_models
        )

        def neg_map(w: np.ndarray) -> float:
            w = np.abs(w) / np.abs(w).sum()
            fused = ScoreFuser(
                weights={n: float(wi) for n, wi in zip(names, w)}
            ).fuse(val_score_dict)
            return -self._map3(fused, labels)

        result = minimize(
            neg_map, x0=init_weights, method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-5},
        )

        opt = np.abs(result.x)
        opt /= opt.sum()
        weight_dict = {n: float(w) for n, w in zip(names, opt)}
        logger.info(f"Optimal weights : {weight_dict}")
        logger.info(f"Ensemble MAP@3  : {-result.fun:.4f}")
        return weight_dict

    # ------------------------------------------------------------------
    # single-method ensemble
    # ------------------------------------------------------------------

    def ensemble(
        self,
        val_score_dict:  Dict[str, np.ndarray],
        test_score_dict: Dict[str, np.ndarray],
        val_labels:      List[str],
        method:          str = "weighted_score",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run one ensemble strategy.

        Returns
        -------
        (val_fused, test_fused, val_map3)
        """
        logger.info(f"Running ensemble: {method}")

        if method == "weighted_score":
            val_cal, temperatures = self.calibrate_all(val_score_dict, val_labels)
            weights   = self.find_optimal_weights(val_cal, val_labels)
            # transfer calibration to test (same T, no fitting)
            test_cal  = {
                name: scores / temperatures.get(name, 1.0)
                for name, scores in test_score_dict.items()
            }
            fuser      = ScoreFuser(weights=weights)
            val_fused  = fuser.fuse(val_cal)
            test_fused = fuser.fuse(test_cal)

        elif method == "rank_average":
            val_fused  = self.rank_avg.average_ranks(val_score_dict)
            test_fused = self.rank_avg.average_ranks(test_score_dict)

        elif method == "soft_vote":
            val_fused  = np.mean(
                [self._softmax(s) for s in val_score_dict.values()], axis=0
            )
            test_fused = np.mean(
                [self._softmax(s) for s in test_score_dict.values()], axis=0
            )

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        val_map3 = self._map3(val_fused, val_labels)
        logger.info(f"[{method}] MAP@3 = {val_map3:.4f}")
        return val_fused, test_fused, val_map3

    # ------------------------------------------------------------------
    # auto-select best method
    # ------------------------------------------------------------------

    def run_all_methods_and_select_best(
        self,
        val_score_dict:  Dict[str, np.ndarray],
        test_score_dict: Dict[str, np.ndarray],
        val_labels:      List[str],
    ) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """
        Try every strategy; return the best by validation MAP@3.

        Returns
        -------
        (best_val_fused, best_test_fused, best_method, best_map3)
        """
        methods  = ["weighted_score", "rank_average", "soft_vote"]
        best_map3        = 0.0
        best_val_fused:  Optional[np.ndarray] = None
        best_test_fused: Optional[np.ndarray] = None
        best_method      = ""
        rows: List[Dict] = []

        for method in methods:
            try:
                vf, tf, map3 = self.ensemble(
                    val_score_dict, test_score_dict, val_labels, method
                )
                rows.append({"Method": method, "MAP@3": f"{map3:.4f}"})
                if map3 > best_map3:
                    best_map3, best_val_fused, best_test_fused, best_method = (
                        map3, vf, tf, method
                    )
            except Exception as exc:
                logger.warning(f"[{method}] failed: {exc}")

        # print comparison table
        df_res = pd.DataFrame(rows)
        sep    = "─" * 40
        print(f"\n{sep}")
        print("  Ensemble Method Comparison")
        print(sep)
        print(df_res.to_string(index=False))
        print(sep)
        print(f"  Best: {best_method}  MAP@3={best_map3:.4f}")
        print(sep)
        logger.info(f"Best ensemble: {best_method}  MAP@3={best_map3:.4f}")

        # W&B logging
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "final/best_ensemble_method": best_method,
                    "final/best_ensemble_map3":   best_map3,
                })
                tbl = wandb.Table(
                    columns=["Method", "MAP@3"],
                    data=[[r["Method"], float(r["MAP@3"])] for r in rows],
                )
                wandb.log({"ensemble/method_comparison": tbl})
        except Exception:
            pass

        return best_val_fused, best_test_fused, best_method, best_map3