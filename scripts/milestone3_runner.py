# milestone3_runner.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


#: Names of the four RAG score artefacts produced / consumed by M3.
_RAG_ARTEFACT_NAMES: List[str] = [
    "rag_vote_val.npy",
    "rag_vote_test.npy",
    "rag_semantic_val.npy",
    "rag_semantic_test.npy",
]


# ─────────────────────────────────────────────────────────────────────────────
# Artifact helpers  (thin wrappers around cfg.artifact_* methods)
# ─────────────────────────────────────────────────────────────────────────────

def _all_rag_artifacts_exist(cfg) -> bool:
    """
    True when every RAG artefact is available from either
    cfg.artifacts_load_dir (pre-built Kaggle input) or
    cfg.artifacts_save_dir (locally saved).
    """
    return all(cfg.artifact_exists(name) for name in _RAG_ARTEFACT_NAMES)


def _load_rag_artifacts(cfg) -> Optional[Dict[str, np.ndarray]]:
    """
    Load all four RAG .npy artefacts via cfg.load_artifact().

    Search order per file (handled inside Config)
    ─────────────────────────────────────────────
    1. cfg.artifacts_load_dir / name  (pre-built Kaggle input artifact)
    2. cfg.artifacts_save_dir / name  (locally saved artifact)

    Returns
    -------
    dict  – keyed by stem name (without extension) if ALL four files load OK.
    None  – if any file is missing or fails to load.
    """
    if not _all_rag_artifacts_exist(cfg):
        return None

    loaded: Dict[str, np.ndarray] = {}
    for name in _RAG_ARTEFACT_NAMES:
        try:
            loaded[name.replace(".npy", "")] = cfg.load_artifact(name)
        except Exception as exc:
            logger.warning(
                f"[artefact] Found {name} but failed to load: {exc}"
            )
            return None          # all-or-nothing

    return loaded


def _save_rag_artifacts(
    cfg,
    rag_vote_val     : np.ndarray,
    rag_vote_test    : np.ndarray,
    rag_semantic_val : np.ndarray,
    rag_semantic_test: np.ndarray,
) -> None:
    """
    Save all four RAG score arrays via cfg.save_artifact()
    (always writes to cfg.artifacts_save_dir).
    """
    payload = {
        "rag_vote_val.npy"     : rag_vote_val,
        "rag_vote_test.npy"    : rag_vote_test,
        "rag_semantic_val.npy" : rag_semantic_val,
        "rag_semantic_test.npy": rag_semantic_test,
    }
    for name, array in payload.items():
        cfg.save_artifact(array, name)

    logger.info(
        f"[M3] RAG artefacts saved → {cfg.artifacts_save_dir}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_milestone3(
    train_df       : pd.DataFrame,
    val_df         : pd.DataFrame,
    test_df        : pd.DataFrame,
    retrieval_model: str = "all-mpnet-base-v2",
    top_k          : int = 5,
) -> Dict[str, Any]:
    """
    Execute the full Milestone 3 RAG pipeline.

    Artefact loading priority
    ─────────────────────────
    1. cfg.artifacts_load_dir  – pre-computed artefacts from a previous
                                 Kaggle notebook run (read-only mount).
    2. cfg.artifacts_save_dir  – artefacts saved by a previous local run.
    3. Compute from scratch and save to cfg.artifacts_save_dir.

    Parameters
    ----------
    train_df        : full training DataFrame (used as retrieval corpus)
    val_df          : validation split
    test_df         : test DataFrame
    retrieval_model : Sentence-BERT model name for FAISS index
    top_k           : neighbours to retrieve per query

    Returns
    -------
    results dict with keys:
        "rag_vote_val", "rag_vote_test",
        "rag_semantic_val", "rag_semantic_test",
        "rag_combined_val", "rag_combined_test",
        "val_contexts", "test_contexts", "train_contexts",
        "metrics"  (ablation MAP@3 values)
    """
    from config.config import Config
    cfg = Config()

    from scripts.milestone2_runner import _Evaluator
    evaluator = _Evaluator()

    option_cols = cfg.options
    results: Dict[str, Any] = {"metrics": {}}

    logger.info(
        f"[M3] artifacts_load_dir : {cfg.artifacts_load_dir} "
        f"(exists={cfg.artifacts_load_dir.exists()})"
    )
    logger.info(f"[M3] artifacts_save_dir : {cfg.artifacts_save_dir}")

    # ── Step 1: Try to load cached artefacts ─────────────────────────────────
    #
    #   Config.load_artifact() checks artifacts_load_dir first (pre-built
    #   Kaggle input), then falls back to artifacts_save_dir (local cache).
    #   _load_rag_artifacts() wraps this with an all-or-nothing guard so we
    #   never use a partial set of stale score arrays.
    #
    artefacts: Optional[Dict[str, np.ndarray]] = _load_rag_artifacts(cfg)

    # ── Step 2: Use cached artefacts OR compute ───────────────────────────────
    if artefacts is not None:
        logger.info("[M3] Reusing RAG artefacts from artifact directories.")

        rag_vote_val      = artefacts["rag_vote_val"]
        rag_vote_test     = artefacts["rag_vote_test"]
        rag_semantic_val  = artefacts["rag_semantic_val"]
        rag_semantic_test = artefacts["rag_semantic_test"]

    else:
        logger.info("[M3] No cached artefacts found – computing from scratch …")

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

        # ── Persist to artifacts_save_dir ─────────────────────────────────
        _save_rag_artifacts(
            cfg,
            rag_vote_val,
            rag_vote_test,
            rag_semantic_val,
            rag_semantic_test,
        )

        # Context strings are cheap to compute; store them now.
        results["val_contexts"]   = rag.get_retrieval_context(val_df)
        results["test_contexts"]  = rag.get_retrieval_context(test_df)
        results["train_contexts"] = rag.get_retrieval_context(train_df)
        if results["val_contexts"]:
            logger.info(
                f"[M3] Sample context: {results['val_contexts'][0][:200]} …"
            )

    # ── Step 3: Populate results ──────────────────────────────────────────────
    results["rag_vote_val"]      = rag_vote_val
    results["rag_vote_test"]     = rag_vote_test
    results["rag_semantic_val"]  = rag_semantic_val
    results["rag_semantic_test"] = rag_semantic_test

    rag_combined_val             = rag_vote_val  + rag_semantic_val
    rag_combined_test            = rag_vote_test + rag_semantic_test
    results["rag_combined_val"]  = rag_combined_val
    results["rag_combined_test"] = rag_combined_test

    # ── Step 4: Context strings (if not yet populated from fresh compute) ─────
    if "val_contexts" not in results:
        logger.info(
            "[M3] Generating retrieval context strings "
            "(artefacts were cached – rebuilding index for context only) …"
        )
        rag = RAGPipeline(
            cfg,
            retrieval_model_name=retrieval_model,
            top_k_retrieve=top_k,
        )
        rag.build_index(train_df)
        results["val_contexts"]   = rag.get_retrieval_context(val_df)
        results["test_contexts"]  = rag.get_retrieval_context(test_df)
        results["train_contexts"] = rag.get_retrieval_context(train_df)

    # ── Step 5: Ablation study ────────────────────────────────────────────────
    results["metrics"].update(
        _run_ablation(
            val_df      = val_df,
            vote_scores = rag_vote_val,
            sem_scores  = rag_semantic_val,
            combined    = rag_combined_val,
            evaluator   = evaluator,
            cfg         = cfg,
            option_cols = option_cols,
        )
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation(
    val_df      : pd.DataFrame,
    vote_scores : np.ndarray,
    sem_scores  : np.ndarray,
    combined    : np.ndarray,
    evaluator,
    cfg,
    option_cols : List[str],
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