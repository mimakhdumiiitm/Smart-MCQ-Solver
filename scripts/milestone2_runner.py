# milestone2_runner.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.transformer_ranker import (
    TransformerEmbeddingRanker,
    ZeroShotMCQRanker,
    _load_or_none,
    _scores_exist,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Inline minimal Evaluator (no external dependency needed)
# ─────────────────────────────────────────────────────────────────────────────

class _Evaluator:
    """
    Self-contained evaluator so the notebook needs zero extra imports.
    """

    def scores_to_top_k_predictions(
        self,
        scores:      np.ndarray,       # (n_samples, n_options)
        option_cols: List[str],
        k:           int = 3,
    ) -> List[List[str]]:
        """Return top-k option labels (e.g. ['A','C','B']) per sample."""
        preds = []
        for row in scores:
            top_idx = np.argsort(row)[::-1][:k]
            preds.append([option_cols[i] for i in top_idx])
        return preds

    def evaluate(
        self,
        df:     pd.DataFrame,
        preds:  List[List[str]],
        cfg,
        split:  str = "val",
    ) -> Dict[str, float]:
        """Compute MAP@3. Expects df to have an 'answer' column."""
        if "answer" not in df.columns:
            logger.warning("No 'answer' column found – MAP@3 set to 0.")
            return {f"{split}/map@3": 0.0}

        aps = []
        for pred_list, true in zip(preds, df["answer"].tolist()):
            score, hits = 0.0, 0
            for rank, p in enumerate(pred_list, 1):
                if p == str(true):
                    hits  += 1
                    score += hits / rank
            aps.append(score / min(1, 1))   # MAP@K with 1 relevant doc

        map3 = float(np.mean(aps))
        return {f"{split}/map@3": map3}


# ─────────────────────────────────────────────────────────────────────────────
# Auto data-loader
# ─────────────────────────────────────────────────────────────────────────────

def _auto_load_data(
    cfg,
    val_df:  Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    If DataFrames are not passed, try to build them from cfg paths.
    Applies minimal cleaning (adds *_clean columns) if missing.
    """

    def _ensure_clean_cols(df: pd.DataFrame, option_cols: List[str]) -> pd.DataFrame:
        """Add *_clean columns if they don't already exist."""
        if "prompt_clean" not in df.columns:
            prompt_col = next(
                (c for c in ["prompt", "question", "Question"] if c in df.columns),
                df.columns[0],
            )
            df["prompt_clean"] = df[prompt_col].astype(str).str.strip()

        for opt in option_cols:
            if f"{opt}_clean" not in df.columns and opt in df.columns:
                df[f"{opt}_clean"] = df[opt].astype(str).str.strip()

        return df

    option_cols = cfg.options

    # ── val_df ────────────────────────────────────────────────────────────────
    if val_df is None:
        # Try processed CSV first, then raw train file
        processed = cfg.output_dir / "train_processed.csv"
        raw       = cfg.data_dir   / cfg.train_file

        if processed.exists():
            logger.info(f"Loading val_df from {processed}")
            val_df = pd.read_csv(processed)
        elif raw.exists():
            logger.info(f"Loading val_df from {raw} (raw train file)")
            full   = pd.read_csv(raw)
            # simple 80/20 split used as val
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        else:
            raise FileNotFoundError(
                f"Cannot find validation data. "
                f"Tried: {processed}, {raw}\n"
                f"Pass val_df explicitly or set cfg.data_dir / cfg.train_file correctly."
            )

    if test_df is None:
        processed = cfg.output_dir / "test_processed.csv"
        raw       = cfg.data_dir   / cfg.test_file

        if processed.exists():
            logger.info(f"Loading test_df from {processed}")
            test_df = pd.read_csv(processed)
        elif raw.exists():
            logger.info(f"Loading test_df from {raw}")
            test_df = pd.read_csv(raw)
        else:
            logger.warning("No test file found – using val_df as test placeholder.")
            test_df = val_df.copy()

    val_df  = _ensure_clean_cols(val_df,  option_cols)
    test_df = _ensure_clean_cols(test_df, option_cols)

    return val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Internal pipeline helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _run_zero_shot(cfg, val_df, test_df):
    try:
        ranker      = ZeroShotMCQRanker(cfg, model_key=cfg.zs_model_key, batch_size=cfg.batch_size)
        val_scores  = ranker.predict_scores(val_df)
        test_scores = ranker.predict_scores(test_df)
        ranker.free()
        return val_scores, test_scores
    except Exception as exc:
        logger.error(f"ZeroShotMCQRanker failed: {exc}")
        dummy = np.zeros((len(val_df),  len(cfg.options)))
        return dummy, np.zeros((len(test_df), len(cfg.options)))


def _run_transformer_embeddings(cfg, val_df, test_df):
    try:
        ranker      = TransformerEmbeddingRanker(
            cfg, model_key=cfg.tr_model_key,
            batch_size=cfg.batch_size, max_length=cfg.max_length,
        )
        val_scores  = ranker.predict_scores(val_df)
        test_scores = ranker.predict_scores(test_df)
        ranker.free()
        return val_scores, test_scores
    except Exception as exc:
        logger.error(f"TransformerEmbeddingRanker failed: {exc}")
        dummy = np.zeros((len(val_df),  len(cfg.options)))
        return dummy, np.zeros((len(test_df), len(cfg.options)))


def _load_baseline_scores(out_dir, provided):
    if provided:
        return provided
    scores  = {}
    mapping = {
        "tfidf_val": "tfidf_val_scores.npy",
        "w2v_val":   "w2v_val_scores.npy",
        "sbert_val": "sbert_val_scores.npy",
    }
    for key, filename in mapping.items():
        arr = _load_or_none(out_dir / filename)
        if arr is not None:
            scores[key] = arr
        else:
            logger.warning(f"[M2] Baseline file not found: {filename}")
    return scores


def _print_comparison_table(transformer_metrics, baseline_scores, evaluator, val_df, cfg, option_cols):
    rows = []

    for tag, key in [("TF-IDF", "tfidf_val"), ("Word2Vec", "w2v_val"), ("SBERT", "sbert_val")]:
        if key in baseline_scores:
            preds  = evaluator.scores_to_top_k_predictions(baseline_scores[key], option_cols)
            metric = evaluator.evaluate(val_df, preds, cfg, split=f"{tag}_compare")
            map3   = metric.get(f"{tag}_compare/map@3", 0.0)
        else:
            map3 = float("nan")
        rows.append({"Method": tag, "Stage": "Phase 1", "MAP@3": map3})

    rows.append({
        "Method": "Zero-shot DeBERTa NLI",
        "Stage":  "Milestone 2",
        "MAP@3":  transformer_metrics.get("zeroshot_val/map@3", float("nan")),
    })
    rows.append({
        "Method": "DeBERTa Embedding (cross-enc.)",
        "Stage":  "Milestone 2",
        "MAP@3":  transformer_metrics.get("transformer_val/map@3", float("nan")),
    })

    df_table = pd.DataFrame(rows).sort_values("MAP@3", ascending=False)
    df_table["MAP@3"] = df_table["MAP@3"].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

    sep = "─" * 55
    print(f"\n{sep}")
    print("  MAP@3 Comparison – Phase 1 vs Milestone 2")
    print(sep)
    print(df_table.to_string(index=False))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC entry point  ←  THE ONLY THING YOU CALL FROM THE NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────

def run_milestone2(
    cfg              = None,
    val_df           : Optional[pd.DataFrame]        = None,
    test_df          : Optional[pd.DataFrame]        = None,
    evaluator                                        = None,
    option_cols      : Optional[List[str]]           = None,
    baseline_scores  : Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Execute the full Milestone 2 pipeline.

    All parameters are OPTIONAL – sane defaults are applied automatically.

    Minimal notebook usage
    ----------------------
    >>> from src.milestone2_runner import run_milestone2
    >>> results = run_milestone2()

    With custom config
    ------------------
    >>> from src.config import Config
    >>> cfg = Config(data_dir="/kaggle/input/my-dataset")
    >>> results = run_milestone2(cfg)

    With pre-loaded DataFrames
    --------------------------
    >>> results = run_milestone2(cfg, val_df=my_val, test_df=my_test)

    Parameters
    ----------
    cfg             : Config object (auto-created with defaults if None)
    val_df          : validation DataFrame (auto-loaded from disk if None)
    test_df         : test DataFrame (auto-loaded from disk if None)
    evaluator       : evaluator with evaluate() / scores_to_top_k_predictions()
                      (built-in _Evaluator used if None)
    option_cols     : e.g. ["A","B","C","D","E"]  (taken from cfg if None)
    baseline_scores : Phase-1 score arrays (loaded from .npy files if None)

    Returns
    -------
    dict with keys
        "zs_val_scores", "zs_test_scores",
        "transformer_val_scores", "transformer_test_scores",
        "metrics"
    """

    # ── 0. Build defaults for anything not supplied ───────────────────────────
    if cfg is None:
        from src.config import Config          # lazy import – works without config.py too
        cfg = Config()
        logger.info(f"[M2] Using default Config | device={cfg.device} | out={cfg.output_dir}")

    if evaluator is None:
        evaluator = _Evaluator()
        logger.info("[M2] Using built-in _Evaluator")

    if option_cols is None:
        option_cols = cfg.options

    # ── 1. Load / build DataFrames ────────────────────────────────────────────
    val_df, test_df = _auto_load_data(cfg, val_df, test_df)

    out                = cfg.output_dir
    results            : Dict[str, Any] = {"metrics": {}}

    # ── 2. Load Phase-1 baseline scores ──────────────────────────────────────
    baseline_scores = _load_baseline_scores(out, baseline_scores)

    # ── 3. Zero-shot NLI ranker ───────────────────────────────────────────────
    zs_val_path  = out / "zs_val_scores.npy"
    zs_test_path = out / "zs_test_scores.npy"

    if _scores_exist(zs_val_path, zs_test_path):
        logger.info("[M2] Reusing cached zero-shot scores.")
        zs_val_scores  = np.load(zs_val_path)
        zs_test_scores = np.load(zs_test_path)
    else:
        logger.info("[M2] Running ZeroShotMCQRanker …")
        zs_val_scores, zs_test_scores = _run_zero_shot(cfg, val_df, test_df)
        np.save(zs_val_path,  zs_val_scores)
        np.save(zs_test_path, zs_test_scores)
        logger.info(f"[M2] Zero-shot scores saved → {out}")

    zs_preds   = evaluator.scores_to_top_k_predictions(zs_val_scores, option_cols)
    zs_metrics = evaluator.evaluate(val_df, zs_preds, cfg, split="zeroshot_val")
    results["metrics"].update(zs_metrics)
    results["zs_val_scores"]  = zs_val_scores
    results["zs_test_scores"] = zs_test_scores
    print(f"[M2] Zero-shot MAP@3 = {zs_metrics.get('zeroshot_val/map@3', 0):.4f}")

    # ── 4. Transformer embedding ranker ──────────────────────────────────────
    tr_val_path  = out / "transformer_val_scores.npy"
    tr_test_path = out / "transformer_test_scores.npy"

    if _scores_exist(tr_val_path, tr_test_path):
        logger.info("[M2] Reusing cached transformer embedding scores.")
        transformer_val_scores  = np.load(tr_val_path)
        transformer_test_scores = np.load(tr_test_path)
    else:
        logger.info("[M2] Running TransformerEmbeddingRanker …")
        transformer_val_scores, transformer_test_scores = _run_transformer_embeddings(
            cfg, val_df, test_df
        )
        np.save(tr_val_path,  transformer_val_scores)
        np.save(tr_test_path, transformer_test_scores)
        logger.info(f"[M2] Transformer scores saved → {out}")

    tr_preds   = evaluator.scores_to_top_k_predictions(transformer_val_scores, option_cols)
    tr_metrics = evaluator.evaluate(val_df, tr_preds, cfg, split="transformer_val")
    results["metrics"].update(tr_metrics)
    results["transformer_val_scores"]  = transformer_val_scores
    results["transformer_test_scores"] = transformer_test_scores
    print(f"[M2] Transformer MAP@3 = {tr_metrics.get('transformer_val/map@3', 0):.4f}")

    # ── 5. Comparison table ───────────────────────────────────────────────────
    _print_comparison_table(
        results["metrics"], baseline_scores,
        evaluator, val_df, cfg, option_cols,
    )

    return results