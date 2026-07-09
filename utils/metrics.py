# utils/metrics.py
# updated for milestone 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from typing import List, Dict, Union

from config.config import (
    TOP_K,
    OPTION_COLS,
    COLORS,
    RESULTS_PLOT_PATH,
)


# ==================================================================
# EXISTING FUNCTIONS — UNCHANGED
# ==================================================================

def precision_at_k(predicted: list,
                   actual: str,
                   k: int) -> float:
    """
    Compute Precision@k for a single question with one correct answer.

    Usage:
        score = precision_at_k(["B", "A", "C"], "B", k=1)   # 1.0
    """
    if not predicted or not actual:
        return 0.0
    top_k = predicted[:k]
    hits  = sum(1 for p in top_k if p == actual)
    return hits / k


def average_precision_at_k(predicted: list,
                            actual: str,
                            k: int = TOP_K) -> float:
    """
    Compute Average Precision@k for a single question.

    Usage:
        ap = average_precision_at_k(["B", "A", "C"], "B", k=3)  # 1.0
        ap = average_precision_at_k(["A", "B", "C"], "B", k=3)  # 0.5
        ap = average_precision_at_k(["A", "C", "B"], "B", k=3)  # 0.333
        ap = average_precision_at_k(["A", "C", "D"], "B", k=3)  # 0.0
    """
    if not predicted or not actual:
        return 0.0

    num_relevant = 1
    ap           = 0.0
    hits         = 0

    for i in range(1, k + 1):
        if i > len(predicted):
            break
        if predicted[i - 1] == actual:
            hits += 1
            ap   += hits / i

    return ap / min(num_relevant, k)


def map_at_k(predictions: List[list],
             actuals: List[str],
             k: int = TOP_K) -> float:
    """
    Compute Mean Average Precision at k (MAP@k) across all questions.

    Usage:
        score = map_at_k(
            predictions=[["B","A","C"], ["A","B","C"]],
            actuals=["B", "B"],
            k=3
        )
    """
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(actuals)} actuals."
        )

    ap_scores = [
        average_precision_at_k(pred, actual, k)
        for pred, actual in zip(predictions, actuals)
    ]
    return float(np.mean(ap_scores))


def parse_prediction_string(pred_str: str) -> list:
    """
    Parse a space-separated prediction string into a list of labels.

    Usage:
        labels = parse_prediction_string("B A C")
        # ["B", "A", "C"]
    """
    return str(pred_str).strip().split()


def parse_predictions_series(series: pd.Series) -> list:
    """
    Parse a pandas Series of prediction strings into a list of lists.

    Usage:
        preds = parse_predictions_series(submission_df["Prediction"])
    """
    return [parse_prediction_string(s) for s in series]


def evaluation_report(predictions: List[list],
                       actuals: List[str],
                       k: int = TOP_K) -> pd.DataFrame:
    """
    Build a per-question evaluation report with AP@k and rank.

    Usage:
        report = evaluation_report(preds, actuals, k=3)
    """
    records = []
    for idx, (pred, actual) in enumerate(zip(predictions, actuals)):
        ap   = average_precision_at_k(pred, actual, k)
        rank = None
        for r, p in enumerate(pred[:k], start=1):
            if p == actual:
                rank = r
                break

        row = {"question_idx": idx, "actual": actual}
        for i in range(1, k + 1):
            row[f"pred_{i}"] = pred[i - 1] if i <= len(pred) else "-"

        row["correct_rank"] = rank if rank is not None else f">{k}"
        row["ap_at_k"]      = round(ap, 4)
        records.append(row)

    report    = pd.DataFrame(records)
    map_score = report["ap_at_k"].mean()
    print(f"MAP@{k} = {map_score:.4f}")
    return report


def rank_distribution(predictions: List[list],
                       actuals: List[str],
                       k: int = TOP_K) -> dict:
    """
    Count how many times the correct answer appeared at each rank.

    Usage:
        dist = rank_distribution(preds, actuals, k=3)
    """
    distribution = {r: 0 for r in range(1, k + 1)}
    distribution["not_found"] = 0

    for pred, actual in zip(predictions, actuals):
        found = False
        for r, p in enumerate(pred[:k], start=1):
            if p == actual:
                distribution[r] += 1
                found = True
                break
        if not found:
            distribution["not_found"] += 1

    return distribution


def compare_strategies(strategies: dict,
                        actuals: List[str],
                        k: int = TOP_K) -> pd.DataFrame:
    """
    Compare MAP@k scores across multiple prediction strategies.

    Usage:
        results = compare_strategies(
            strategies={"random": random_preds, "tfidf": tfidf_preds},
            actuals=train_df["answer"].tolist()
        )
    """
    records = []
    for name, preds in strategies.items():
        score = map_at_k(preds, actuals, k)
        records.append({"strategy": name, f"map_at_{k}": round(score, 4)})
        print(f"  {name:<25} MAP@{k} = {score:.4f}")

    return pd.DataFrame(records).sort_values(
        f"map_at_{k}", ascending=False
    ).reset_index(drop=True)


def format_submission(ids: list,
                       predictions: List[list],
                       k: int = TOP_K) -> pd.DataFrame:
    """
    Format predictions into the competition submission format.

    Usage:
        sub_df = format_submission(test_df["id"].tolist(), preds, k=3)
    """
    pred_strings = [" ".join(p[:k]) for p in predictions]
    return pd.DataFrame({
        "ID"        : ids,
        "Prediction": pred_strings,
    })


# ==================================================================
# NEW: MAP3EVALUATOR CLASS  (Transformer pipeline — Cell 12)
# ==================================================================

class MAP3Evaluator:
    """
    Computes MAP@3 metrics for the transformer pipeline results
    and logs them to Weights & Biases.

    Accepts the results list produced by EnsembleRanker.predict_all()
    which contains per-record dicts with "top_labels", "answer",
    and "combined_scores" keys.

    Usage:
        evaluator     = MAP3Evaluator()
        train_metrics = evaluator.compute_map3(train_results)
        strategy_df   = evaluator.compare_strategies(
            train_data, emb_scores_list, zs_scores_list, ranker
        )
        evaluator.plot_and_log(train_metrics, strategy_df, train_results)
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

    # ------------------------------------------------------------------
    # COMPUTE MAP@3
    # ------------------------------------------------------------------

    def compute_map3(self, results: List[dict]) -> dict:
        """
        Compute MAP@3 and hit-rate metrics from a results list.

        Iterates over records that have a non-null "answer" field
        (train set). Records without answers (test set) are skipped.

        Args:
            results : list of result dicts from EnsembleRanker.predict_all()

        Returns:
            dict with keys:
                MAP@3           : mean average precision at 3
                num_questions   : number of evaluated questions
                hit@1           : fraction with correct answer at rank 1
                hit@2           : fraction with correct in top 2
                hit@3           : fraction with correct in top 3
                correct_at_pos1 : count of rank-1 hits
                correct_at_pos2 : count of rank-2 hits
                correct_at_pos3 : count of rank-3 hits

        Usage:
            metrics = evaluator.compute_map3(train_results)
            print(metrics["MAP@3"])
        """
        ap_scores     = []
        position_hits = {1: 0, 2: 0, 3: 0}

        for res in results:
            true_ans = res.get("answer")
            if not true_ans:
                continue

            preds = res["top_labels"]
            ap    = average_precision_at_k(preds, true_ans, k=self.top_k)
            ap_scores.append(ap)

            for pos, pred in enumerate(preds[:self.top_k], start=1):
                if pred == true_ans:
                    position_hits[pos] += 1

        n    = len(ap_scores)
        map3 = float(np.mean(ap_scores)) if ap_scores else 0.0

        return {
            "MAP@3"          : round(map3, 4),
            "num_questions"  : n,
            "hit@1"          : round(position_hits[1] / n, 4) if n else 0,
            "hit@2"          : round(
                                   (position_hits[1] + position_hits[2]) / n, 4
                               ) if n else 0,
            "hit@3"          : round(sum(position_hits.values()) / n, 4) if n else 0,
            "correct_at_pos1": position_hits[1],
            "correct_at_pos2": position_hits[2],
            "correct_at_pos3": position_hits[3],
        }

    # ------------------------------------------------------------------
    # COMPARE STRATEGIES
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        records        : List[dict],
        emb_scores_list: List[dict],
        zs_scores_list : List[dict],
        ranker,
    ) -> pd.DataFrame:
        """
        Evaluate MAP@3 for each scoring strategy independently.

        Strategies evaluated: "embedding", "zeroshot", "ensemble"

        Args:
            records         : list of record dicts
            emb_scores_list : list of embedding score dicts
            zs_scores_list  : list of zero-shot score dicts
            ranker          : EnsembleRanker instance

        Returns:
            pd.DataFrame with columns:
                [Strategy, MAP@3, num_questions, hit@1, hit@2, hit@3, ...]

        Usage:
            strategy_df = evaluator.compare_strategies(
                train_data, emb_scores_list, zs_scores_list, ranker
            )
        """
        strategies = ["embedding", "zeroshot", "ensemble"]
        rows       = []

        for strat in strategies:
            # Generate fresh predictions for this strategy
            strat_results = ranker.predict_all(
                records, emb_scores_list, zs_scores_list
            )
            # Override top_labels using this strategy
            for res, emb_s, zs_s in zip(
                strat_results, emb_scores_list, zs_scores_list
            ):
                res["top_labels"] = ranker.predict_one(
                    emb_s, zs_s, strategy=strat
                )

            metrics = self.compute_map3(strat_results)
            rows.append({"Strategy": strat, **metrics})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # PLOT AND LOG TO W&B
    # ------------------------------------------------------------------

    def plot_and_log(
        self,
        metrics     : dict,
        strategy_df : pd.DataFrame,
        results     : List[dict],
        plot_path   : str = RESULTS_PLOT_PATH,
    ) -> None:
        """
        Generate evaluation plots and log everything to W&B.

        Three subplots:
            1. MAP@3 bar chart per strategy
            2. Hit rate by rank position (ensemble)
            3. Combined score distribution: correct vs incorrect options

        Args:
            metrics     : output of compute_map3() for the ensemble strategy
            strategy_df : output of compare_strategies()
            results     : list of result dicts from EnsembleRanker.predict_all()
            plot_path   : file path to save the PNG figure

        Usage:
            evaluator.plot_and_log(train_metrics, strategy_df, train_results)
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "MCQ Solver - Transformer Pipeline Results",
            fontsize=14, fontweight="bold"
        )

        # ── Plot 1: Strategy Comparison ────────────────────────────
        ax1  = axes[0]
        bars = ax1.bar(
            strategy_df["Strategy"],
            strategy_df["MAP@3"],
            color     = ["#4C72B0", "#DD8452", "#55A868"],
            edgecolor = "black",
            linewidth = 0.8,
        )
        for bar, val in zip(bars, strategy_df["MAP@3"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=10,
            )
        ax1.set_title("MAP@3 by Strategy")
        ax1.set_ylabel("MAP@3")
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)

        # ── Plot 2: Hit Rate by Rank Position ──────────────────────
        ax2        = axes[1]
        positions  = ["Hit@1", "Hit@2", "Hit@3"]
        hit_values = [metrics["hit@1"], metrics["hit@2"], metrics["hit@3"]]
        ax2.bar(
            positions, hit_values,
            color     = ["#2ecc71", "#3498db", "#9b59b6"],
            edgecolor = "black",
            linewidth = 0.8,
        )
        for i, val in enumerate(hit_values):
            ax2.text(
                i, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
        ax2.set_title("Hit Rate by Position (Ensemble)")
        ax2.set_ylabel("Hit Rate")
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)

        # ── Plot 3: Score Distribution ──────────────────────────────
        ax3             = axes[2]
        correct_scores  = []
        incorrect_scores = []

        for res in results:
            true_ans = res.get("answer")
            if not true_ans:
                continue
            for label, score in res["combined_scores"].items():
                if label == true_ans:
                    correct_scores.append(score)
                else:
                    incorrect_scores.append(score)

        ax3.hist(
            correct_scores,   bins=20, alpha=0.7,
            label="Correct",  color="#2ecc71"
        )
        ax3.hist(
            incorrect_scores, bins=20, alpha=0.7,
            label="Incorrect", color="#e74c3c"
        )
        ax3.set_title("Score Distribution")
        ax3.set_xlabel("Ensemble Score")
        ax3.set_ylabel("Count")
        ax3.legend()
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.show()

        # ── W&B Logging ────────────────────────────────────────────
        wandb.log({
            **{f"train/{k}": v for k, v in metrics.items()},
            "strategy_comparison": wandb.Table(dataframe=strategy_df),
            "results_plot"       : wandb.Image(plot_path),
        })
        print("Metrics logged to W&B")