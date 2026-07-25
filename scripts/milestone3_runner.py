# milestone3_runner.py


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_none(path: Path):
    if path.exists():
        logger.info(f"[cache] Reusing {path.name}")
        return np.load(path)
    return None


def _all_rag_cached(out: Path) -> bool:
    names = [
        "rag_vote_val.npy", "rag_vote_test.npy",
        "rag_semantic_val.npy", "rag_semantic_test.npy",
    ]
    return all((out / n).exists() for n in names)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_milestone3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    retrieval_model: str = "all-mpnet-base-v2",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Execute the full Milestone 3 RAG pipeline.

    Parameters
    ----------
    cfg            : Config object
    train_df       : full training DataFrame (used as retrieval corpus)
    val_df         : validation split
    test_df        : test DataFrame
    evaluator      : evaluator module / object
    option_cols    : e.g. ["A","B","C","D","E"]
    retrieval_model: Sentence-BERT model name for FAISS index
    top_k          : neighbours to retrieve per query

    Returns
    -------
    results dict with keys:
        "rag_vote_val", "rag_vote_test",
        "rag_semantic_val", "rag_semantic_test",
        "rag_combined_val", "rag_combined_test",
        "val_contexts", "test_contexts", "train_contexts",
        "metrics"  (ablation MAP@3 values)
    """
    if cfg is None:
        from config.config import Config
        cfg = Config()

    if evaluator is None:
        from scripts.milestone2_runner import _Evaluator
        evaluator = _Evaluator()

    if option_cols is None:
        option_cols = cfg.options
        
    out = cfg.output_dir
    results: Dict[str, Any] = {"metrics": {}}

    # ── Step 1: Load cached scores or compute fresh ───────────────────────────
    if _all_rag_cached(out):
        logger.info("[M3] All RAG score files found – loading from cache.")
        rag_vote_val      = np.load(out / "rag_vote_val.npy")
        rag_vote_test     = np.load(out / "rag_vote_test.npy")
        rag_semantic_val  = np.load(out / "rag_semantic_val.npy")
        rag_semantic_test = np.load(out / "rag_semantic_test.npy")

    else:
        logger.info("[M3] Building RAG pipeline …")
        rag = RAGPipeline(
            cfg,
            retrieval_model_name=retrieval_model,
            top_k_retrieve=top_k,
        )
        rag.build_index(train_df)

        rag_vote_val      = rag.compute_rag_scores(val_df)
        rag_vote_test     = rag.compute_rag_scores(test_df)
        rag_semantic_val  = rag.compute_semantic_context_scores(val_df)
        rag_semantic_test = rag.compute_semantic_context_scores(test_df)

        # ── Persist ──────────────────────────────────────────────────────────
        np.save(out / "rag_vote_val.npy",      rag_vote_val)
        np.save(out / "rag_vote_test.npy",     rag_vote_test)
        np.save(out / "rag_semantic_val.npy",  rag_semantic_val)
        np.save(out / "rag_semantic_test.npy", rag_semantic_test)
        logger.info(f"[M3] RAG scores saved to {out}")

        # Context strings (not cached – cheap to recompute)
        results["val_contexts"]   = rag.get_retrieval_context(val_df)
        results["test_contexts"]  = rag.get_retrieval_context(test_df)
        results["train_contexts"] = rag.get_retrieval_context(train_df)
        if results["val_contexts"]:
            logger.info(
                f"[M3] Sample context: {results['val_contexts'][0][:200]} …"
            )

    # Populate results dict
    results["rag_vote_val"]      = rag_vote_val
    results["rag_vote_test"]     = rag_vote_test
    results["rag_semantic_val"]  = rag_semantic_val
    results["rag_semantic_test"] = rag_semantic_test

    # Combined
    rag_combined_val              = rag_vote_val  + rag_semantic_val
    rag_combined_test             = rag_vote_test + rag_semantic_test
    results["rag_combined_val"]  = rag_combined_val
    results["rag_combined_test"] = rag_combined_test

    # ── Step 2: Context strings (if not yet populated) ────────────────────────
    if "val_contexts" not in results:
        logger.info("[M3] Generating retrieval context strings from cached scores …")
        # Need to rebuild rag object just for context (lightweight)
        rag = RAGPipeline(cfg, retrieval_model_name=retrieval_model, top_k_retrieve=top_k)
        rag.build_index(train_df)
        results["val_contexts"]   = rag.get_retrieval_context(val_df)
        results["test_contexts"]  = rag.get_retrieval_context(test_df)
        results["train_contexts"] = rag.get_retrieval_context(train_df)

    # ── Step 3: Ablation study ────────────────────────────────────────────────
    results["metrics"].update(
        _run_ablation(
            val_df       = val_df,
            vote_scores  = rag_vote_val,
            sem_scores   = rag_semantic_val,
            combined     = rag_combined_val,
            evaluator    = evaluator,
            cfg          = cfg,
            option_cols  = option_cols,
        )
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation(
    val_df:      pd.DataFrame,
    vote_scores: np.ndarray,
    sem_scores:  np.ndarray,
    combined:    np.ndarray,
    evaluator,
    cfg,
    option_cols: List[str],
) -> Dict[str, float]:
    """
    Compare three RAG scoring variants on the validation set.

    Returns a flat dict of MAP@3 values.
    """
    ablation_metrics: Dict[str, float] = {}
    variants = [
        ("rag_vote_only",     vote_scores),
        ("rag_semantic_only", sem_scores),
        ("rag_combined",      combined),
    ]

    rows: List[Dict[str, Any]] = []

    for split_name, scores in variants:
        preds   = evaluator.scores_to_top_k_predictions(scores, option_cols)
        metrics = evaluator.evaluate(val_df, preds, cfg, split=split_name)
        map3    = metrics.get(f"{split_name}/map@3", 0.0)
        ablation_metrics[f"{split_name}/map@3"] = map3
        rows.append({"RAG Variant": split_name, "MAP@3": f"{map3:.4f}"})
        logger.info(f"[M3] {split_name} MAP@3 = {map3:.4f}")

    # ── print ablation table ──────────────────────────────────────────────────
    df_abl = pd.DataFrame(rows)
    sep    = "─" * 45
    logger.info(f"\n{sep}")
    logger.info("  Milestone 3 – RAG Ablation Study")
    logger.info(sep)
    logger.info(df_abl.to_string(index=False))
    logger.info(sep)
    print(df_abl.to_string(index=False))

    return ablation_metrics