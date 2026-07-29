# scripts/milestone5_runner.py
"""
Ensemble runner – Milestone 4 (Ensembling week).

Loads pre-computed .npy score artefacts produced by M1/M2/M3,
runs all ensemble strategies, logs to W&B, and saves the best
ensemble scores to cfg.output_dir.

Expected artefacts in cfg.output_dir
--------------------------------------
tfidf_val.npy         tfidf_test.npy
w2v_val.npy           w2v_test.npy
sbert_val.npy         sbert_test.npy        (optional but recommended)
rag_combined_val.npy  rag_combined_test.npy (from M3, optional)

Nothing from previous milestones is re-run.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping: internal model name → W&B job_type tag
# wandb_init.py accepts: "tfidf", "word2vec", "sbert"
# ---------------------------------------------------------------------------
_MODEL_TAG_MAP: Dict[str, str] = {
    "tfidf"       : "tfidf",
    "w2v"         : "word2vec",
    "sbert"       : "sbert",
    "rag_combined": "sbert",   # RAG uses SBERT internally; closest proxy tag
}

# Human-readable run names that will appear in the W&B dashboard
_MODEL_RUN_NAME: Dict[str, str] = {
    "tfidf"       : "ensemble-tfidf",
    "w2v"         : "ensemble-word2vec",
    "sbert"       : "ensemble-sbert",
    "rag_combined": "ensemble-rag-combined",
}


# ---------------------------------------------------------------------------
# Artefact helpers
# ---------------------------------------------------------------------------

def _load_score_pair(
    directory: Path,
    val_file : str,
    test_file: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a (val, test) score pair – returns None if either file is missing."""
    vp, tp = directory / val_file, directory / test_file
    if vp.exists() and tp.exists():
        return np.load(vp), np.load(tp)
    logger.warning(f"[artefact] Missing file(s): {val_file} or {test_file} in {directory}")
    return None


def _load_all_scores(
    out: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Discover and load every available model score pair.

    Returns
    -------
    val_scores  : {model_name: (n_val,  n_options)}
    test_scores : {model_name: (n_test, n_options)}
    """
    candidates = {
        "tfidf"       : ("tfidf_val.npy",       "tfidf_test.npy"),
        "w2v"         : ("w2v_val.npy",          "w2v_test.npy"),
        "sbert"       : ("sbert_val.npy",        "sbert_test.npy"),
        "rag_combined": ("rag_combined_val.npy", "rag_combined_test.npy"),
    }

    val_scores:  Dict[str, np.ndarray] = {}
    test_scores: Dict[str, np.ndarray] = {}

    for name, (vf, tf) in candidates.items():
        pair = _load_score_pair(out, vf, tf)
        if pair is not None:
            val_scores[name], test_scores[name] = pair
            logger.info(f"[artefact] Loaded scores: {name}  "
                        f"val={val_scores[name].shape}  "
                        f"test={test_scores[name].shape}")

    return val_scores, test_scores


# ---------------------------------------------------------------------------
# Classification metrics helper
# ---------------------------------------------------------------------------

def _compute_all_metrics(
    scores     : np.ndarray,
    val_labels : List[str],
    evaluator  : Any,
    option_cols: List[str],
) -> Dict[str, float]:
    """
    Compute the five required W&B metrics from a (n_samples, n_options)
    score matrix.

    Required keys (must match REQUIRED_METRICS in wandb_init.py):
        f1_score, accuracy, precision, recall, map_at_k

    Top-1 prediction drives the classification metrics;
    top-3 predictions drive MAP@3.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    # ── top-1 predictions (for classification metrics) ────────────────────
    top1_indices = np.argmax(scores, axis=1)
    top1_preds   = [option_cols[i] for i in top1_indices]

    # ── top-3 predictions (for MAP@3) ─────────────────────────────────────
    top3_preds = evaluator.scores_to_top_k_predictions(scores, option_cols)
    map3       = evaluator.mean_average_precision_at_k(val_labels, top3_preds)

    return {
        "accuracy" : float(accuracy_score(val_labels, top1_preds)),
        "f1_score" : float(
            f1_score(val_labels, top1_preds, average="macro", zero_division=0)
        ),
        "precision": float(
            precision_score(val_labels, top1_preds, average="macro", zero_division=0)
        ),
        "recall"   : float(
            recall_score(val_labels, top1_preds, average="macro", zero_division=0)
        ),
        "map_at_k" : float(map3),
    }


# ---------------------------------------------------------------------------
# W&B per-model logging
# Uses the project's existing init_wandb / log_model_metrics / finish_run
# ---------------------------------------------------------------------------

def _log_per_model_wandb(
    val_scores : Dict[str, np.ndarray],
    val_labels : List[str],
    evaluator  : Any,
    option_cols: List[str],
    cfg        : Any,
) -> Dict[str, Dict[str, float]]:
    """
    Open one dedicated W&B run per model, log the five required metrics,
    then close the run.

    Satisfies:
        ✅ All models have a valid W&B run
        ✅ ≥ 3 runs compared with common metrics (f1_score, accuracy, …)

    Returns
    -------
    per_model_metrics : {model_name: {metric_name: value}}
    """
    # Import the project-level helpers – do NOT redefine them
    from utils.wandb_init import init_wandb, log_model_metrics, finish_run

    per_model_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, scores in val_scores.items():
        tag      = _MODEL_TAG_MAP.get(model_name, "sbert")
        run_name = _MODEL_RUN_NAME.get(model_name, f"ensemble-{model_name}")

        # ── compute metrics before opening the run (keeps run time short) ──
        metrics = _compute_all_metrics(
            scores, val_labels, evaluator, option_cols
        )
        per_model_metrics[model_name] = metrics

        logger.info(
            f"[W&B] {model_name} → tag={tag}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_score']:.4f}  "
            f"map@3={metrics['map_at_k']:.4f}"
        )

        # ── open run ──────────────────────────────────────────────────────
        run = init_wandb(
            config   = cfg,
            run_name = run_name,
            model_tag= tag,
        )

        # ── log all five required metrics ─────────────────────────────────
        # log_model_metrics already warns if any key from REQUIRED_METRICS
        # is missing, so passing the full dict is all we need.
        log_model_metrics(run, metrics)

        # ── additional structured logging for richer W&B UI ───────────────
        #    (histograms, per-class breakdown, score distribution)
        if run is not None:
            try:
                import wandb

                # Score distribution per option column
                for idx, col in enumerate(option_cols):
                    run.log({
                        f"score_dist/{col}": wandb.Histogram(scores[:, idx])
                    })

                # Per-label accuracy breakdown
                from sklearn.metrics import classification_report
                top1_preds = [
                    option_cols[i] for i in np.argmax(scores, axis=1)
                ]
                report = classification_report(
                    val_labels, top1_preds,
                    output_dict=True,
                    zero_division=0,
                )
                for label, label_metrics in report.items():
                    if isinstance(label_metrics, dict):
                        for metric_name, value in label_metrics.items():
                            run.log({
                                f"per_class/{label}/{metric_name}": value
                            })

            except Exception as exc:
                # Never crash the run on optional rich logging
                logger.warning(f"[W&B rich logging] {model_name}: {exc}")

        # ── close run ─────────────────────────────────────────────────────
        finish_run(run)

    return per_model_metrics


# ---------------------------------------------------------------------------
# Ensemble W&B run  (one final run for the winning strategy)
# ---------------------------------------------------------------------------

def _log_ensemble_wandb(
    cfg          : Any,
    best_method  : str,
    best_val     : np.ndarray,
    val_labels   : List[str],
    evaluator    : Any,
    option_cols  : List[str],
    all_metrics  : Dict[str, float],
) -> None:
    """
    Log the winning ensemble as its own W&B run so it appears in the
    same project and can be compared against individual model runs.
    """
    from utils.wandb_init import init_wandb, log_model_metrics, finish_run

    metrics = _compute_all_metrics(
        best_val, val_labels, evaluator, option_cols
    )

    # Merge in any extra strategy-level metrics already computed
    merged = {**metrics, **all_metrics}

    run = init_wandb(
        config   = cfg,
        run_name = f"ensemble-best-{best_method}",
        model_tag= "sbert",          # tag must be one of the three known tags
    )

    if run is not None:
        try:
            import wandb

            # Log the five required metrics
            log_model_metrics(run, metrics)

            # Also log every strategy's MAP@3 for a full comparison table
            strategy_data = [
                [k, v]
                for k, v in merged.items()
                if "map" in k.lower()
            ]
            if strategy_data:
                tbl = wandb.Table(
                    columns=["Strategy / Model", "MAP@3"],
                    data=strategy_data,
                )
                run.log({"ensemble/full_comparison": tbl})

            # Log best method as a text summary
            run.log({"ensemble/best_method": best_method})

        except Exception as exc:
            logger.warning(f"[W&B ensemble run] extra logging failed: {exc}")

    finish_run(run)
    logger.info(f"[W&B] Ensemble run logged: ensemble-best-{best_method}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_milestone5(
    val_df : pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Load pre-computed scores → ensemble → save → return results.

    Parameters
    ----------
    val_df  : validation DataFrame  (must contain the answer column)
    test_df : test DataFrame        (shape reference only; no labels needed)

    Returns
    -------
    dict with keys
        ensemble_val     (n_val,  n_options)  np.ndarray
        ensemble_test    (n_test, n_options)  np.ndarray
        ensemble_method  str
        metrics          dict  – MAP@3 per strategy + per individual model
    """
    # ── project imports (nothing re-run, just reusing existing objects) ────
    from config.config import Config
    from src.ensemble.orchestrator import EnsembleOrchestrator

    # Reuse the evaluator that was defined in M2
    from scripts.milestone2_runner import _Evaluator

    cfg         = Config()
    evaluator   = _Evaluator()
    option_cols : List[str] = cfg.options
    answer_col  : str       = getattr(cfg, "answer_col", "answer")
    out         : Path      = Path(cfg.output_dir)
    results     : Dict[str, Any] = {"metrics": {}}

    val_labels: List[str] = val_df[answer_col].tolist()

    # ── 1. Load score artefacts ────────────────────────────────────────────
    val_scores, test_scores = _load_all_scores(out)

    if len(val_scores) < 2:
        raise RuntimeError(
            f"Need at least 2 score sources in {out}. "
            "Run M1 / M2 / M3 first to generate .npy artefacts."
        )

    logger.info(f"[ensemble] Available sources: {list(val_scores.keys())}")

    # ── 2. Per-model W&B runs (satisfies the ≥3 comparable runs rule) ─────
    #       Uses init_wandb → log_model_metrics → finish_run from wandb_init.py
    per_model_metrics = _log_per_model_wandb(
        val_scores, val_labels, evaluator, option_cols, cfg
    )

    # Store individual MAP@3 in the results dict
    for model_name, metrics in per_model_metrics.items():
        key = f"{model_name}/map@3"
        results["metrics"][key] = metrics["map_at_k"]
        logger.info(
            f"[individual] {model_name}  "
            f"MAP@3={metrics['map_at_k']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_score']:.4f}"
        )

    # ── 3. Ensemble (all three strategies, auto-select best) ───────────────
    orchestrator = EnsembleOrchestrator(
        config      = cfg,
        evaluator   = evaluator,
        option_cols = option_cols,
    )

    best_val, best_test, best_method, best_map3 = (
        orchestrator.run_all_methods_and_select_best(
            val_scores, test_scores, val_labels
        )
    )

    results["ensemble_val"]    = best_val
    results["ensemble_test"]   = best_test
    results["ensemble_method"] = best_method
    results["metrics"]["ensemble/map@3"] = best_map3

    # ── 4. Ensemble W&B run ────────────────────────────────────────────────
    _log_ensemble_wandb(
        cfg         = cfg,
        best_method = best_method,
        best_val    = best_val,
        val_labels  = val_labels,
        evaluator   = evaluator,
        option_cols = option_cols,
        all_metrics = results["metrics"],
    )

    # ── 5. Persist ensemble score matrices ────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ensemble_val.npy",  best_val)
    np.save(out / "ensemble_test.npy", best_test)
    logger.info(f"[ensemble] Scores saved → {out}")

    # ── 6. Final console summary ──────────────────────────────────────────
    summary_rows = [
        {"Model / Strategy": k, "MAP@3": f"{v:.4f}"}
        for k, v in results["metrics"].items()
    ]
    df_summary = pd.DataFrame(summary_rows)
    sep = "═" * 52
    print(f"\n{sep}")
    print("  Final Ensemble Summary  (all MAP@3 on validation set)")
    print(sep)
    print(df_summary.to_string(index=False))
    print(sep)
    print(f"  Best ensemble: {best_method}   MAP@3 = {best_map3:.4f}")
    print(sep)

    return results