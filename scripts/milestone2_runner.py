# milestone2_runner.py

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.transformer_ranker import (
    TransformerEmbeddingRanker,
    ZeroShotMCQRanker,
    _compute_metrics,
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
# W&B helpers  (delegate to utils/wandb_init.py where possible)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from utils.wandb_init import (
        authenticate,
        init_wandb,
        log_model_metrics,
        finish_run,
        _WANDB_AVAILABLE,
        REQUIRED_METRICS,
        COMPARABLE_MODELS,
    )
    _WANDB_UTILS_AVAILABLE = True
    logger.info("[W&B] utils/wandb_init imported successfully.")
except ImportError:
    _WANDB_UTILS_AVAILABLE = False
    logger.warning(
        "[W&B] utils/wandb_init not found – falling back to inline helpers."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Non-interactive W&B authentication  (called ONCE before any wandb.init)
# ─────────────────────────────────────────────────────────────────────────────

def _do_wandb_auth(api_key: Optional[str] = None) -> None:
    """
    Authenticate with W&B exactly once, **never** prompting the user.

    Priority
    ────────
    1. ``api_key`` argument (caller-supplied)
    2. ``WANDB_API_KEY`` environment variable
    3. ``authenticate()`` from utils/wandb_init  (Kaggle Secrets → env → netrc)
    4. Existing netrc / previous login  (silent re-use)

    If none of the above succeeds W&B logging is silently disabled via the
    ``WANDB_MODE=disabled`` environment variable so that no interactive
    prompt is ever shown.
    """
    resolved_key: Optional[str] = (
        api_key
        or os.environ.get("WANDB_API_KEY")
    )

    if resolved_key is None and _WANDB_UTILS_AVAILABLE:
        try:
            authenticate()
            resolved_key = os.environ.get("WANDB_API_KEY")
        except Exception as exc:
            logger.debug(f"[W&B] authenticate() returned: {exc}")

    try:
        import wandb

        if resolved_key:
            wandb.login(
                key       = resolved_key,
                relogin   = True,
                anonymous = "never",
            )
            logger.info("[W&B] Logged in with API key (non-interactive).")
        else:
            wandb.login(
                anonymous = "never",
                relogin   = False,
            )
            logger.info("[W&B] Re-using existing W&B credentials.")

    except Exception as exc:
        logger.warning(
            f"[W&B] Silent login failed ({exc}). "
            "Setting WANDB_MODE=disabled – metrics will not be uploaded."
        )
        os.environ["WANDB_MODE"] = "disabled"


# ─────────────────────────────────────────────────────────────────────────────
# Per-run W&B helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wandb_init(
    project    : str,
    run_name   : str,
    config_dict: dict,
    cfg        = None,
    model_tag  : str = "custom",
) -> object:
    """
    Start a W&B run. Auth must have been performed by ``_do_wandb_auth()``
    before this is called – this function never calls ``wandb.login()``.
    """
    if _WANDB_UTILS_AVAILABLE and cfg is not None and model_tag in COMPARABLE_MODELS:
        try:
            run = init_wandb(cfg, run_name=run_name, model_tag=model_tag)
            if run is not None:
                logger.info(f"[W&B] Run started via init_wandb → {run.url}")
            return run
        except Exception as exc:
            logger.warning(
                f"[W&B] init_wandb failed ({exc}). Trying inline fallback."
            )

    try:
        import wandb
        run = wandb.init(
            project = project,
            name    = run_name,
            config  = config_dict,
            reinit  = "finish_previous",
        )
        logger.info(f"[W&B] Run started (inline) → {run.url}")
        return run
    except Exception as exc:
        logger.warning(
            f"[W&B] wandb.init failed ({exc}). Metrics will not be uploaded."
        )
        return None


def _wandb_finish(run) -> None:
    """Safely finish a W&B run (delegates to utils when available)."""
    if run is None:
        return

    if _WANDB_UTILS_AVAILABLE:
        try:
            finish_run(run)
            return
        except Exception as exc:
            logger.warning(f"[W&B] finish_run() failed: {exc}")

    try:
        run.finish()
        logger.info("[W&B] Run finished.")
    except Exception as exc:
        logger.warning(f"[W&B] run.finish() failed: {exc}")


def _wandb_log(run, metrics: dict) -> None:
    """Log metrics to a W&B run (delegates to utils when available)."""
    if run is None:
        return

    if _WANDB_UTILS_AVAILABLE:
        try:
            log_model_metrics(run, metrics)
            return
        except Exception as exc:
            logger.warning(f"[W&B] log_model_metrics failed: {exc}")

    try:
        run.log(metrics)
    except Exception as exc:
        logger.warning(f"[W&B] run.log failed: {exc}")


def _build_required_metrics_payload(raw_metrics: dict) -> dict:
    """
    Map internal metric keys → canonical REQUIRED_METRICS names:
        map_at_3  → map_at_k
        f1_macro  → f1_score
    Also seeds any still-missing REQUIRED_METRICS keys with None.
    """
    mapping = {
        "map_at_3": "map_at_k",
        "f1_macro": "f1_score",
    }
    payload = {mapping.get(k, k): v for k, v in raw_metrics.items()}

    if _WANDB_UTILS_AVAILABLE:
        for req in REQUIRED_METRICS:
            payload.setdefault(req, None)

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Inline minimal Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class _Evaluator:
    """Self-contained evaluator (no external dependency needed)."""

    def scores_to_top_k_predictions(
        self,
        scores      : np.ndarray,
        option_cols : List[str],
        k           : int = 3,
    ) -> List[List[str]]:
        preds = []
        for row in scores:
            top_idx = np.argsort(row)[::-1][:k]
            preds.append([option_cols[i] for i in top_idx])
        return preds

    def evaluate(
        self,
        df    : pd.DataFrame,
        preds : List[List[str]],
        cfg,
        split : str = "val",
    ) -> Dict[str, float]:
        if "answer" not in df.columns:
            logger.warning("No 'answer' column – MAP@3 set to 0.")
            return {f"{split}/map@3": 0.0}

        aps = []
        for pred_list, true in zip(preds, df["answer"].tolist()):
            score, hits = 0.0, 0
            for rank, p in enumerate(pred_list, 1):
                if p == str(true):
                    hits  += 1
                    score += hits / rank
            aps.append(score)

        return {f"{split}/map@3": float(np.mean(aps))}


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_clean_cols(df: pd.DataFrame, option_cols: List[str]) -> pd.DataFrame:
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


def _auto_load_data(
    cfg,
    val_df  : Optional[pd.DataFrame],
    test_df : Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load validation and test DataFrames.

    Search order for each file
    ──────────────────────────
    1. Caller-supplied DataFrame       (skip search entirely)
    2. cfg.processed_dir               (locally processed files)
    3. cfg.data_dir / cfg.*_file       (raw CSV fallback)
    """
    option_cols = cfg.options

    # ── Validation set ────────────────────────────────────────────────────────
    if val_df is None:
        processed_local = cfg.processed_dir / "train_processed.csv"
        raw             = cfg.data_dir / cfg.train_file

        if processed_local.exists():
            logger.info(f"[data] Loading val_df from processed: {processed_local}")
            full   = pd.read_csv(processed_local)
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        elif raw.exists():
            logger.info(f"[data] Loading val_df from raw: {raw}")
            full   = pd.read_csv(raw)
            val_df = full.iloc[int(len(full) * 0.8):].reset_index(drop=True)
        else:
            raise FileNotFoundError(
                "Cannot find validation data. "
                f"Tried processed={processed_local}, raw={raw}"
            )

    # ── Test set ──────────────────────────────────────────────────────────────
    if test_df is None:
        processed_local = cfg.processed_dir / "test_processed.csv"
        raw             = cfg.data_dir / cfg.test_file

        if processed_local.exists():
            logger.info(f"[data] Loading test_df from processed: {processed_local}")
            test_df = pd.read_csv(processed_local)
        elif raw.exists():
            logger.info(f"[data] Loading test_df from raw: {raw}")
            test_df = pd.read_csv(raw)
        else:
            logger.warning("[data] No test file found – using val_df as placeholder.")
            test_df = val_df.copy()

    val_df  = _ensure_clean_cols(val_df,  option_cols)
    test_df = _ensure_clean_cols(test_df, option_cols)

    return val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Baseline score loader  (uses Config artifact helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _load_baseline_scores(
    cfg,
    provided: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Load Phase-1 baseline .npy score arrays via cfg.load_artifact().

    Search order per file (handled inside Config)
    ─────────────────────
    1. Caller-supplied dict           (``provided``)
    2. cfg.artifacts_load_dir / name  (pre-built Kaggle input artifact)
    3. cfg.artifacts_save_dir / name  (locally saved artifact)
    """
    if provided:
        return provided

    scores  : Dict[str, np.ndarray] = {}
    mapping = {
        "tfidf_val": "tfidf_val_scores.npy",
        "w2v_val"  : "w2v_val_scores.npy",
        "sbert_val": "sbert_val_scores.npy",
    }

    for key, filename in mapping.items():
        if cfg.artifact_exists(filename):
            try:
                scores[key] = cfg.load_artifact(filename)
                logger.info(f"[baseline] Loaded {key} ← {filename}")
            except Exception as exc:
                logger.warning(
                    f"[baseline] Found {filename} but failed to load: {exc}"
                )
        else:
            logger.warning(
                f"[baseline] {filename} not found in artifacts_load_dir "
                f"({cfg.artifacts_load_dir}) or artifacts_save_dir "
                f"({cfg.artifacts_save_dir}). Skipping {key}."
            )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Score cache helpers  (uses Config artifact helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_scores(
    cfg      : "Config",
    val_name : str,
    test_name: str,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Try to load a cached pair of score .npy files via cfg.load_artifact().

    Search order (handled inside Config)
    ─────────────────────────────────────
    1. cfg.artifacts_load_dir / name  (pre-built Kaggle input artifact)
    2. cfg.artifacts_save_dir / name  (locally saved artifact)

    Returns ``(val_scores, test_scores)`` on success, ``None`` when neither
    source has **both** files.
    """
    val_exists  = cfg.artifact_exists(val_name)
    test_exists = cfg.artifact_exists(test_name)

    if val_exists and test_exists:
        try:
            val_scores  = cfg.load_artifact(val_name)
            test_scores = cfg.load_artifact(test_name)
            logger.info(
                f"[cache] Reusing cached score arrays: {val_name}, {test_name}"
            )
            return val_scores, test_scores
        except Exception as exc:
            logger.warning(
                f"[cache] Score cache found but failed to load "
                f"({val_name}, {test_name}): {exc}"
            )
            return None

    if val_exists and not test_exists:
        logger.info(
            f"[cache] Found {val_name} but no {test_name} – "
            "will recompute both to stay consistent."
        )

    logger.info(
        f"[cache] Score cache not found ({val_name}, {test_name}) – "
        "will recompute."
    )
    return None


def _save_scores(
    cfg        : "Config",
    val_scores : np.ndarray,
    test_scores: np.ndarray,
    val_name   : str,
    test_name  : str,
) -> None:
    """
    Save score arrays via cfg.save_artifact()
    (always writes to artifacts_save_dir).
    """
    cfg.save_artifact(val_scores,  val_name)
    cfg.save_artifact(test_scores, test_name)
    logger.info(
        f"[cache] Saved score artifacts to {cfg.artifacts_save_dir}: "
        f"{val_name}, {test_name}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_zero_shot(cfg, val_df, test_df, wandb_run):
    try:
        ranker = ZeroShotMCQRanker(
            cfg,
            model_key  = cfg.zs_model_key,
            batch_size = cfg.batch_size,
            wandb_run  = wandb_run,
        )
        val_scores  = ranker.predict_scores(val_df)
        test_scores = ranker.predict_scores(test_df)
        ranker.free()
        return val_scores, test_scores
    except Exception as exc:
        logger.error(f"ZeroShotMCQRanker failed: {exc}")
        dummy = np.zeros((len(val_df), len(cfg.options)))
        return dummy, np.zeros((len(test_df), len(cfg.options)))


def _run_transformer_embeddings(cfg, val_df, test_df, wandb_run):
    try:
        ranker = TransformerEmbeddingRanker(
            cfg,
            model_key  = cfg.tr_model_key,
            batch_size = cfg.batch_size,
            max_length = cfg.max_length,
            wandb_run  = wandb_run,
        )
        val_scores  = ranker.predict_scores(val_df)
        test_scores = ranker.predict_scores(test_df)
        ranker.free()
        return val_scores, test_scores
    except Exception as exc:
        logger.error(f"TransformerEmbeddingRanker failed: {exc}")
        dummy = np.zeros((len(val_df), len(cfg.options)))
        return dummy, np.zeros((len(test_df), len(cfg.options)))


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(all_metrics: Dict[str, dict]) -> None:
    rows = []
    for method, m in all_metrics.items():
        rows.append(
            {
                "Method"    : method,
                "MAP@3"     : m.get("map_at_3",  float("nan")),
                "Accuracy"  : m.get("accuracy",  float("nan")),
                "F1 (macro)": m.get("f1_macro",  float("nan")),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values("MAP@3", ascending=False)
        .reset_index(drop=True)
    )
    for col in ["MAP@3", "Accuracy", "F1 (macro)"]:
        df[col] = df[col].map(
            lambda x: f"{x:.4f}"
            if not (isinstance(x, float) and np.isnan(x))
            else "N/A"
        )

    sep = "─" * 68
    print(f"\n{sep}")
    print("  MAP@3 / Accuracy / F1  –  All methods compared")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_milestone2(
    cfg             = None,
    val_df          : Optional[pd.DataFrame]           = None,
    test_df         : Optional[pd.DataFrame]           = None,
    evaluator                                          = None,
    option_cols     : Optional[List[str]]              = None,
    baseline_scores : Optional[Dict[str, np.ndarray]] = None,
    wandb_project   : str                              = "smart-mcq-solver",
    wandb_api_key   : Optional[str]                   = None,
) -> Dict[str, Any]:

    # ── 0. Authenticate ONCE – never interactive ──────────────────────────────
    _do_wandb_auth(wandb_api_key)

    # ── 1. Config / evaluator defaults ───────────────────────────────────────
    if cfg is None:
        from config.config import Config
        cfg = Config()
        logger.info(f"[M2] device={cfg.device} | out={cfg.output_dir}")

    if evaluator is None:
        evaluator = _Evaluator()

    if option_cols is None:
        option_cols = cfg.options

    logger.info(
        f"[M2] artifacts_load_dir : {cfg.artifacts_load_dir} "
        f"(exists={cfg.artifacts_load_dir.exists()})"
    )
    logger.info(
        f"[M2] artifacts_save_dir : {cfg.artifacts_save_dir}"
    )

    # ── 2. Load data ──────────────────────────────────────────────────────────
    val_df, test_df = _auto_load_data(cfg, val_df, test_df)
    logger.info(f"[M2] val={len(val_df)} rows | test={len(test_df)} rows")

    results            : Dict[str, Any]  = {"metrics": {}}
    all_method_metrics : Dict[str, dict] = {}

    # ── 3. Phase-1 baseline scores ────────────────────────────────────────────
    # NOTE: signature changed – cfg is now the first arg (no out_dir needed)
    baseline_scores = _load_baseline_scores(cfg, baseline_scores)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN 1 : Zero-Shot NLI
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  W&B RUN 1 – Zero-Shot NLI Ranker")
    logger.info("═" * 60)

    zs_run = _wandb_init(
        project     = wandb_project,
        run_name    = "zero_shot_nli",
        config_dict = {
            "model_type"  : "zero_shot_nli",
            "model_key"   : cfg.zs_model_key,
            "batch_size"  : cfg.batch_size,
            "n_val"       : len(val_df),
            "n_test"      : len(test_df),
            "option_cols" : option_cols,
        },
        cfg       = cfg,
        model_tag = "zero_shot_nli",
    )

    try:
        cached = _try_load_scores(
            cfg,
            "zs_val_scores.npy",
            "zs_test_scores.npy",
        )

        if cached is not None:
            zs_val_scores, zs_test_scores = cached
            logger.info("[M2] Zero-shot scores loaded from cache/artifact.")
            _wandb_log(zs_run, {"cache_hit": True})
        else:
            logger.info("[M2] Running ZeroShotMCQRanker from scratch …")
            zs_val_scores, zs_test_scores = _run_zero_shot(
                cfg, val_df, test_df, zs_run
            )
            _save_scores(
                cfg,
                zs_val_scores,
                zs_test_scores,
                "zs_val_scores.npy",
                "zs_test_scores.npy",
            )
            _wandb_log(zs_run, {"cache_hit": False})

        zs_metrics         = _compute_metrics(zs_val_scores, val_df, option_cols)
        zs_required        = _build_required_metrics_payload(zs_metrics)
        zs_required_tagged = {f"zeroshot_val/{k}": v for k, v in zs_required.items()}

        _wandb_log(zs_run, zs_required_tagged)

        results["metrics"].update(
            {f"zeroshot_val/{k}": v for k, v in zs_metrics.items()}
        )
        results["zs_val_scores"]  = zs_val_scores
        results["zs_test_scores"] = zs_test_scores
        all_method_metrics["Zero-Shot NLI"] = zs_metrics

        print(
            f"[M2] Zero-Shot  MAP@3={zs_metrics['map_at_3']:.4f} "
            f"| Acc={zs_metrics['accuracy']:.4f} "
            f"| F1={zs_metrics['f1_macro']:.4f}"
        )
    finally:
        _wandb_finish(zs_run)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN 2 : Transformer Embedding Ranker
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  W&B RUN 2 – Transformer Embedding Ranker")
    logger.info("═" * 60)

    tr_run = _wandb_init(
        project     = wandb_project,
        run_name    = "transformer_embed",
        config_dict = {
            "model_type"  : "transformer_embedding",
            "model_key"   : cfg.tr_model_key,
            "batch_size"  : cfg.batch_size,
            "max_length"  : cfg.max_length,
            "n_val"       : len(val_df),
            "n_test"      : len(test_df),
            "option_cols" : option_cols,
        },
        cfg       = cfg,
        model_tag = "transformer_embedding",
    )

    try:
        cached = _try_load_scores(
            cfg,
            "transformer_val_scores.npy",
            "transformer_test_scores.npy",
        )

        if cached is not None:
            transformer_val_scores, transformer_test_scores = cached
            logger.info("[M2] Transformer scores loaded from cache/artifact.")
            _wandb_log(tr_run, {"cache_hit": True})
        else:
            logger.info("[M2] Running TransformerEmbeddingRanker from scratch …")
            transformer_val_scores, transformer_test_scores = (
                _run_transformer_embeddings(cfg, val_df, test_df, tr_run)
            )
            _save_scores(
                cfg,
                transformer_val_scores,
                transformer_test_scores,
                "transformer_val_scores.npy",
                "transformer_test_scores.npy",
            )
            _wandb_log(tr_run, {"cache_hit": False})

        tr_metrics         = _compute_metrics(transformer_val_scores, val_df, option_cols)
        tr_required        = _build_required_metrics_payload(tr_metrics)
        tr_required_tagged = {f"transformer_val/{k}": v for k, v in tr_required.items()}

        _wandb_log(tr_run, tr_required_tagged)

        results["metrics"].update(
            {f"transformer_val/{k}": v for k, v in tr_metrics.items()}
        )
        results["transformer_val_scores"]  = transformer_val_scores
        results["transformer_test_scores"] = transformer_test_scores
        all_method_metrics["Transformer Embed"] = tr_metrics

        print(
            f"[M2] Transformer  MAP@3={tr_metrics['map_at_3']:.4f} "
            f"| Acc={tr_metrics['accuracy']:.4f} "
            f"| F1={tr_metrics['f1_macro']:.4f}"
        )
    finally:
        _wandb_finish(tr_run)

    # ─────────────────────────────────────────────────────────────────────────
    # RUN 3 : Phase-1 Baselines comparison
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  W&B RUN 3 – Phase-1 Baseline Comparison")
    logger.info("═" * 60)

    baseline_tag_map = {
        "tfidf_val": ("TF-IDF",   "tfidf"),
        "w2v_val"  : ("Word2Vec", "word2vec"),
        "sbert_val": ("SBERT",    "sbert"),
    }

    wandb_table_rows: List[list] = []

    for key, (display_tag, model_tag) in baseline_tag_map.items():
        if key not in baseline_scores:
            logger.warning(f"[baseline] {key} scores not available – skipping.")
            continue

        bl_run = _wandb_init(
            project     = wandb_project,
            run_name    = f"baseline_{model_tag}",
            config_dict = {
                "model_type"      : "phase1_baseline",
                "model_tag"       : model_tag,
                "baselines_found" : list(baseline_scores.keys()),
                "n_val"           : len(val_df),
                "option_cols"     : option_cols,
            },
            cfg       = cfg,
            model_tag = model_tag,
        )

        try:
            bm          = _compute_metrics(baseline_scores[key], val_df, option_cols)
            bm_required = _build_required_metrics_payload(bm)
            _wandb_log(bl_run, bm_required)

            tagged = {f"{display_tag}_val/{k}": v for k, v in bm.items()}
            results["metrics"].update(tagged)
            all_method_metrics[display_tag] = bm

            wandb_table_rows.append(
                [display_tag, bm["map_at_3"], bm["accuracy"], bm["f1_macro"]]
            )

            print(
                f"[M2] {display_tag:<10} MAP@3={bm['map_at_3']:.4f} "
                f"| Acc={bm['accuracy']:.4f} "
                f"| F1={bm['f1_macro']:.4f}"
            )
        finally:
            _wandb_finish(bl_run)

    # ── summary W&B Table ─────────────────────────────────────────────────────
    bl_summary_run = _wandb_init(
        project     = wandb_project,
        run_name    = "baseline_compare_summary",
        config_dict = {
            "model_type"      : "phase1_baselines_summary",
            "baselines_found" : list(baseline_scores.keys()),
            "n_val"           : len(val_df),
            "option_cols"     : option_cols,
        },
        cfg       = cfg,
        model_tag = "summary",
    )

    try:
        try:
            import wandb as _wandb
            for method, m in all_method_metrics.items():
                if method not in {t for t, _ in baseline_tag_map.values()}:
                    wandb_table_rows.append(
                        [method, m["map_at_3"], m["accuracy"], m["f1_macro"]]
                    )
            tbl = _wandb.Table(
                columns = ["Method", "MAP@3", "Accuracy", "F1_macro"],
                data    = wandb_table_rows,
            )
            if bl_summary_run is not None:
                bl_summary_run.log({"comparison_table": tbl})
                logger.info("[W&B] Comparison table logged.")
        except Exception as exc:
            logger.warning(f"[W&B] Could not log comparison table: {exc}")
    finally:
        _wandb_finish(bl_summary_run)

    # ── 6. Print final comparison table ──────────────────────────────────────
    _print_comparison_table(all_method_metrics)

    results["all_method_metrics"] = all_method_metrics
    return results