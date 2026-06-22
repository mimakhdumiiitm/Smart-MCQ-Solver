
# metrics.py
# MAP@K and related evaluation metric implementations

import numpy as np
import pandas as pd
from typing import List, Union

from config.config import TOP_K, OPTION_COLS


# ------------------------------------------------------------------
# CORE MAP@K FUNCTIONS
# ------------------------------------------------------------------

def precision_at_k(predicted: list,
                   actual: str,
                   k: int) -> float:
    """
    Compute Precision@k for a single question with one correct answer.

    Precision@k = (number of correct answers in top-k predictions) / k

    Since there is exactly one correct answer:
        = 1/k  if correct answer appears anywhere in predicted[:k]
        = 0    otherwise

    Args:
        predicted : list of predicted option labels, e.g. ["B", "A", "C"]
        actual    : the single correct option label, e.g. "B"
        k         : cutoff rank

    Returns:
        float precision score.

    Usage:
        score = precision_at_k(["B", "A", "C"], "B", k=1)   # 1.0
        score = precision_at_k(["A", "B", "C"], "B", k=1)   # 0.0
        score = precision_at_k(["A", "B", "C"], "B", k=2)   # 0.5
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

    For single-answer MCQ:
        AP@k = sum_{i=1}^{k} [ P@i * rel(i) ] / min(|relevant|, k)

    Where:
        rel(i) = 1 if prediction at position i equals actual answer
        P@i    = precision at position i
        |relevant| = 1 (exactly one correct answer)

    Args:
        predicted : ranked list of predicted option labels
        actual    : the correct option label
        k         : maximum rank to consider

    Returns:
        AP@k score as a float.

    Usage:
        ap = average_precision_at_k(["B", "A", "C"], "B", k=3)  # 1.0
        ap = average_precision_at_k(["A", "B", "C"], "B", k=3)  # 0.5
        ap = average_precision_at_k(["A", "C", "B"], "B", k=3)  # 0.333
        ap = average_precision_at_k(["A", "C", "D"], "B", k=3)  # 0.0
    """
    if not predicted or not actual:
        return 0.0

    num_relevant = 1   # exactly one correct answer
    ap           = 0.0
    hits         = 0

    for i in range(1, k + 1):
        if i > len(predicted):
            break
        if predicted[i - 1] == actual:
            hits += 1
            ap   += hits / i    # P@i when a hit occurs at position i

    return ap / min(num_relevant, k)


def map_at_k(predictions: List[list],
             actuals: List[str],
             k: int = TOP_K) -> float:
    """
    Compute Mean Average Precision at k (MAP@k) across all questions.

    MAP@k = (1/N) * sum_{i=1}^{N} AP@k(i)

    Args:
        predictions : list of ranked prediction lists
                      e.g. [["B","A","C"], ["A","C","D"], ...]
        actuals     : list of correct answer labels
                      e.g. ["B", "A", ...]
        k           : cutoff rank (default from config: TOP_K = 3)

    Returns:
        MAP@k score as a float in [0, 1].

    Usage:
        score = map_at_k(
            predictions=[["B","A","C"], ["A","B","C"]],
            actuals=["B", "B"],
            k=3
        )
        # 0.75
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


# ------------------------------------------------------------------
# PARSING PREDICTIONS
# ------------------------------------------------------------------

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
        # [["B","A","C"], ["A","C","D"], ...]
    """
    return [parse_prediction_string(s) for s in series]


# ------------------------------------------------------------------
# DETAILED EVALUATION REPORT
# ------------------------------------------------------------------

def evaluation_report(predictions: List[list],
                       actuals: List[str],
                       k: int = TOP_K) -> pd.DataFrame:
    """
    Build a per-question evaluation report with AP@k and rank of
    the correct answer.

    Returns:
        DataFrame with columns:
            question_idx, actual, pred_1, pred_2, pred_3,
            correct_rank, ap_at_k

    Usage:
        report = evaluation_report(preds, actuals, k=3)
        print(report)
        print("MAP@3 :", report["ap_at_k"].mean())
    """
    records = []
    for idx, (pred, actual) in enumerate(zip(predictions, actuals)):
        ap   = average_precision_at_k(pred, actual, k)

        # Find rank of correct answer (1-indexed, None if not in top-k)
        rank = None
        for r, p in enumerate(pred[:k], start=1):
            if p == actual:
                rank = r
                break

        row = {
            "question_idx" : idx,
            "actual"       : actual,
        }
        for i in range(1, k + 1):
            row[f"pred_{i}"] = pred[i - 1] if i <= len(pred) else "-"

        row["correct_rank"] = rank if rank is not None else f">{k}"
        row["ap_at_k"]      = round(ap, 4)
        records.append(row)

    report = pd.DataFrame(records)
    map_score = report["ap_at_k"].mean()
    print(f"MAP@{k} = {map_score:.4f}")
    return report


# ------------------------------------------------------------------
# RANK DISTRIBUTION ANALYSIS
# ------------------------------------------------------------------

def rank_distribution(predictions: List[list],
                       actuals: List[str],
                       k: int = TOP_K) -> dict:
    """
    Count how many times the correct answer appeared at each rank
    position (1, 2, 3, ..., or not found).

    Returns:
        dict mapping rank -> count, plus "not_found" key.

    Usage:
        dist = rank_distribution(preds, actuals, k=3)
        for rank, count in dist.items():
            print(rank, count)
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


# ------------------------------------------------------------------
# STRATEGY COMPARISON
# ------------------------------------------------------------------

def compare_strategies(strategies: dict,
                        actuals: List[str],
                        k: int = TOP_K) -> pd.DataFrame:
    """
    Compare MAP@k scores across multiple prediction strategies.

    Args:
        strategies : dict mapping strategy name -> list of prediction lists
                     e.g. {"random": [[...], ...], "tfidf": [[...], ...]}
        actuals    : list of correct answer labels
        k          : evaluation cutoff

    Returns:
        DataFrame with columns ['strategy', 'map_score'] sorted descending.

    Usage:
        results = compare_strategies(
            strategies={
                "random"      : random_preds,
                "tfidf_cosine": tfidf_preds,
                "w2v_cosine"  : w2v_preds,
            },
            actuals=train_df["answer"].tolist()
        )
        print(results)
    """
    records = []
    for name, preds in strategies.items():
        score = map_at_k(preds, actuals, k)
        records.append({"strategy": name, f"map_at_{k}": round(score, 4)})
        print(f"  {name:<25} MAP@{k} = {score:.4f}")

    return pd.DataFrame(records).sort_values(
        f"map_at_{k}", ascending=False
    ).reset_index(drop=True)


# ------------------------------------------------------------------
# FORMAT PREDICTIONS FOR SUBMISSION
# ------------------------------------------------------------------

def format_submission(ids: list,
                       predictions: List[list],
                       k: int = TOP_K) -> pd.DataFrame:
    """
    Format predictions into the competition submission format.

    Args:
        ids         : list of question IDs
        predictions : list of ranked prediction lists
        k           : number of predictions per question

    Returns:
        DataFrame with columns ['ID', 'Prediction'].

    Usage:
        sub_df = format_submission(test_df["id"].tolist(), preds, k=3)
        sub_df.to_csv("submission.csv", index=False)
    """
    pred_strings = [" ".join(p[:k]) for p in predictions]
    return pd.DataFrame({
        "ID"        : ids,
        "Prediction": pred_strings
    })