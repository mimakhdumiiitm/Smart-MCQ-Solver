# milestone3_runner.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

#: Names of the four RAG score artefacts produced / consumed by M3.
_RAG_ARTEFACT_NAMES: List[str] = [
    "rag_vote_val.npy",
    "rag_vote_test.npy",
    "rag_semantic_val.npy",
    "rag_semantic_test.npy",
]


def _try_load_from_dir(directory: Optional[Path]) -> Optional[Dict[str, np.ndarray]]:
    """
    Attempt to load all four RAG .npy artefacts from *directory*.

    Returns
    -------
    dict  – keyed by stem name (without extension) if ALL four files exist.
    None  – if the directory is None / missing or any file is absent.
    """
    if directory is None:
        return None

    directory = Path(directory)
    if not directory.exists():
        logger.debug(f"[artefact] Directory not found: {directory}")
        return None

    loaded: Dict[str, np.ndarray] = {}
    for name in _RAG_ARTEFACT_NAMES:
        p = directory / name
        if not p.exists():
            logger.debug(f"[artefact] Missing in {directory}: {name}")
            return None          # all-or-nothing
        loaded[name.replace(".npy", "")] = np.load(p)
        logger.info(f"[artefact] Loaded {name} from {directory}")

    return loaded


def _all_rag_cached(directory: Path) -> bool:
    """True when every artefact file exists inside *directory*."""
    return all((directory / n).exists() for n in _RAG_ARTEFACT_NAMES)


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

    Artefact loading priority
    -------------------------
    1. ``cfg.kaggle_artifacts_dir``  – pre-computed artefacts from a previous
                                       Kaggle notebook run (read-only mount).
    2. ``cfg.output_dir``            – artefacts saved by a previous local run.
    3. Compute from scratch and save to ``cfg.output_dir``.

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
    out         = cfg.output_dir
    results: Dict[str, Any] = {"metrics": {}}

    # ── Step 1: Resolve artefact source ──────────────────────────────────────
    #
    #   Priority:
    #     (a) Kaggle read-only mount   → cfg.kaggle_artifacts_dir
    #     (b) Local output dir cache   → cfg.output_dir
    #     (c) Compute from scratch
    #
    artefacts: Optional[Dict[str, np.ndarray]] = None
    artefact_source: str = "compute"

    # (a) Check Kaggle artifacts directory (configured, not hardcoded here)
    kaggle_dir: Optional[Path] = getattr(cfg, "kaggle_artifacts_dir", None)
    if kaggle_dir is not None:
        artefacts = _try_load_from_dir(kaggle_dir)
        if artefacts is not None:
            artefact_source = f"kaggle_artifacts ({kaggle_dir})"

    # (b) Fall back to local output-dir cache
    if artefacts is None and _all_rag_cached(out):
        artefacts = _try_load_from_dir(out)
        if artefacts is not None:
            artefact_source = f"local_cache ({out})"

    # ── Step 2: Use cached artefacts OR compute ───────────────────────────────
    if artefacts is not None:
        logger.info(f"[M3] Reusing RAG artefacts from → {artefact_source}")

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

        # ── Persist to local output dir ───────────────────────────────────
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "rag_vote_val.npy",      rag_vote_val)
        np.save(out / "rag_vote_test.npy",     rag_vote_test)
        np.save(out / "rag_semantic_val.npy",  rag_semantic_val)
        np.save(out / "rag_semantic_test.npy", rag_semantic_test)
        logger.info(f"[M3] RAG artefacts saved → {out}")

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
            "(artefacts were cached, rebuilding index for context only) …"
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