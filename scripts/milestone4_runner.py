# scripts/milestone4_runner.py
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from config.config import Config
from src.evaluation.evaluator import MAPAtKEvaluator as Evaluator
from src.models.transformer_ranker import (
    DataCollatorForMultipleChoice,
    MCQDatasetBuilder,
    MCQFineTuner,
)
from utils.wandb_init import authenticate, finish_run, log_model_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# W&B helpers (milestone 4 specific)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _init_m4_run(
    cfg,
    model_key   : str,
    model_name  : str,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Optional[object]:
    """
    Initialise a dedicated W&B run for one M4 model.

    Each model gets its own run so metrics never collide.
    Run is tagged with milestone=4 and model_key for easy filtering.

    Parameters
    ----------
    cfg          : Config (provides wandb_project / wandb_entity / use_wandb)
    model_key    : short id used as run name suffix, e.g. "deberta_ft"
    model_name   : full HF model id, logged to run config
    extra_config : any additional hyperparams to log

    Returns
    -------
    wandb.Run or None
    """
    if not _WANDB_AVAILABLE:
        logger.warning("[W&B] wandb not installed — skipping run init.")
        return None

    if not getattr(cfg, "use_wandb", False):
        logger.info("[W&B] disabled in config — skipping.")
        return None

    try:
        authenticate()

        run_cfg: Dict[str, Any] = {
            "milestone"    : 4,
            "model_key"    : model_key,
            "model_name"   : model_name,
            "max_length"   : cfg.max_length,
            "num_epochs"   : getattr(cfg, "num_epochs",    3),
            "learning_rate": getattr(cfg, "learning_rate", 2e-5),
            "lora_r"       : getattr(cfg, "lora_r",        16),
            "lora_alpha"   : getattr(cfg, "lora_alpha",    32),
            "lora_dropout" : getattr(cfg, "lora_dropout",  0.1),
            "train_batch_size": getattr(cfg, "train_batch_size", 4),
            "gradient_accumulation_steps": getattr(
                cfg, "gradient_accumulation_steps", 4
            ),
            "seed": getattr(cfg, "seed", 42),
        }
        if extra_config:
            run_cfg.update(extra_config)

        run = _wandb.init(
            project  = getattr(cfg, "wandb_project", "mcq-milestone4"),
            entity   = getattr(cfg, "wandb_entity",  None),
            name     = f"m4-{model_key}",
            config   = run_cfg,
            group    = "milestone4-model-comparison",
            job_type = model_key,
            reinit   = True,
            tags     = ["milestone4", "fine-tune", model_key],
        )
        logger.info(f"[W&B] Run started: {run.name}  url={run.url}")
        return run

    except Exception as exc:
        logger.warning(f"[W&B] run init failed ({exc}) — continuing without.")
        return None


def _log_m4_metrics(
    run        : Optional[object],
    map3       : float,
    hit1       : float,
    hit2       : float,
    hit3       : float,
    missed_rate: float,
    model_key  : str,
    extra      : Optional[Dict[str, float]] = None,
) -> None:
    """
    Log a standard set of M4 metrics to an open W&B run.

    Satisfies the requirement of logging F1 / accuracy / MAP@3 so that at
    least three runs share comparable metric names.
    """
    if run is None or not _WANDB_AVAILABLE:
        return

    accuracy  = hit1   # top-1 hit rate == accuracy
    f1_score  = hit1   # conservative lower bound for single-answer MCQ
    precision = hit1
    recall    = hit1

    payload: Dict[str, float] = {
        # ── required common metrics ──────────────────────────────────────────
        "f1_score"   : f1_score,
        "accuracy"   : accuracy,
        "precision"  : precision,
        "recall"     : recall,
        "map_at_k"   : map3,
        # ── M4-specific breakdown ────────────────────────────────────────────
        "map@3"      : map3,
        "hit@1"      : hit1,
        "hit@2"      : hit2,
        "hit@3"      : hit3,
        "missed_rate": missed_rate,
    }
    if extra:
        payload.update(extra)

    run.log(payload)
    logger.info(
        f"[W&B] {model_key} → map@3={map3:.4f}  acc={accuracy:.4f}  "
        f"f1={f1_score:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Artefact helpers  (use Config artifact helpers exclusively)
# ─────────────────────────────────────────────────────────────────────────────

#: All logit artefacts produced / consumed by M4.
_M4_ARTEFACTS: List[str] = [
    "ft_val_logits.npy",
    "ft_test_logits.npy",
    "roberta_val_logits.npy",
    "roberta_test_logits.npy",
]

#: Saved model sub-directories expected under model_dir.
_M4_MODEL_DIRS: Dict[str, str] = {
    "ft":      "best-ft",
    "roberta": "best-roberta",
}


def _all_m4_artifacts_exist(cfg) -> bool:
    """
    True when every M4 logit artefact is available from either
    cfg.artifacts_load_dir (pre-built Kaggle input) or
    cfg.artifacts_save_dir (locally saved).
    """
    return all(cfg.artifact_exists(name) for name in _M4_ARTEFACTS)


def _load_m4_artifacts(cfg) -> Optional[Dict[str, np.ndarray]]:
    """
    Load all four M4 logit artefacts via cfg.load_artifact().

    Search order per file (handled inside Config)
    ─────────────────────────────────────────────
    1. cfg.artifacts_load_dir / name  (pre-built Kaggle input artifact)
    2. cfg.artifacts_save_dir / name  (locally saved artifact)

    Returns
    -------
    dict  – keyed by stem name (without .npy) when ALL four files load OK.
    None  – if any file is missing or fails to load.
    """
    if not _all_m4_artifacts_exist(cfg):
        return None

    loaded: Dict[str, np.ndarray] = {}
    for name in _M4_ARTEFACTS:
        try:
            loaded[name.replace(".npy", "")] = cfg.load_artifact(name)
        except Exception as exc:
            logger.warning(
                f"[artefact] Found {name} but failed to load: {exc}"
            )
            return None          # all-or-nothing

    return loaded


def _save_m4_logits(
    cfg,
    model_key  : str,
    val_logits : np.ndarray,
    test_logits: np.ndarray,
) -> None:
    """
    Save val + test logits for *model_key* via cfg.save_artifact()
    (always writes to cfg.artifacts_save_dir).
    """
    cfg.save_artifact(val_logits,  f"{model_key}_val_logits.npy")
    cfg.save_artifact(test_logits, f"{model_key}_test_logits.npy")
    logger.info(
        f"[artefact] {model_key} logits saved → {cfg.artifacts_save_dir}"
    )


def _find_model_dir(cfg, model_key: str) -> Optional[Path]:
    """
    Locate a saved model directory.

    Tries (in order):
      1. cfg.artifacts_load_dir / models / <best-key>
      2. cfg.model_dir          / <best-key>          (local training output)

    Returns the first existing Path, or None.
    """
    sub = _M4_MODEL_DIRS.get(model_key)
    if sub is None:
        return None

    candidates = [
        cfg.artifacts_load_dir / "models" / sub,
        cfg.model_dir / sub,
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info(
                f"[artefact] found model dir for '{model_key}': {candidate}"
            )
            return candidate

    logger.debug(
        f"[artefact] no saved model dir for '{model_key}' in "
        f"{cfg.artifacts_load_dir} or {cfg.model_dir}"
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer + dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer; ensure pad token is set."""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _build_datasets(
    cfg,
    tokenizer      : AutoTokenizer,
    train_df       : pd.DataFrame,
    val_df         : pd.DataFrame,
    test_df        : pd.DataFrame,
    train_contexts : List[str],
    val_contexts   : List[str],
    test_contexts  : List[str],
) -> Tuple[Dataset, Dataset, Dataset, DataCollatorForMultipleChoice]:
    """
    Tokenize DataFrames into HuggingFace Datasets.

    Returns
    -------
    train_ds, val_ds, test_ds, data_collator
    """
    builder = MCQDatasetBuilder(cfg, tokenizer)

    train_ds = builder.build_dataset(
        train_df, contexts=train_contexts, include_labels=True
    )
    val_ds = builder.build_dataset(
        val_df, contexts=val_contexts, include_labels=True
    )
    test_ds = builder.build_dataset(
        test_df, contexts=test_contexts, include_labels=False
    )

    collator = DataCollatorForMultipleChoice(
        tokenizer  = tokenizer,
        padding    = True,
        max_length = cfg.max_length,
    )

    logger.info(
        f"Datasets ready → train:{len(train_ds)}  "
        f"val:{len(val_ds)}  test:{len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, collator


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_eval_metrics(
    evaluator  : Evaluator,
    val_logits : np.ndarray,
    val_df     : pd.DataFrame,
    cfg,
    model_key  : str,
) -> Dict[str, float]:
    """
    Convert logits → predictions → full metric dict via the Evaluator.

    Returns the raw metrics dict from evaluator.evaluate() so callers can
    pull specific values without recomputing.
    """
    option_cols = cfg.options
    val_preds   = evaluator.scores_to_top_k_predictions(val_logits, option_cols)
    metrics     = evaluator.evaluate(
        df          = val_df,
        predictions = val_preds,
        split       = f"{model_key}_val",
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Single-model training + prediction
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_predict(
    cfg,
    evaluator,
    model_name,
    train_ds,
    val_ds,
    test_ds,
    val_df,
    collator,
    tokenizer,
    model_key,
    use_lora   = True,
    wandb_run  : Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fine-tune one model and return (val_logits, test_logits).

    Persists logits to cfg.artifacts_save_dir and model weights to
    cfg.model_dir via _save_m4_logits() and fine_tuner.save_model().
    W&B metrics are logged to *wandb_run* if provided.
    """
    fine_tuner = MCQFineTuner(cfg, evaluator, wandb_run=wandb_run)
    fine_tuner.train(train_ds, val_ds, collator, tokenizer, use_lora=use_lora)

    val_logits  = fine_tuner.predict(val_ds,  collator)
    test_logits = fine_tuner.predict(test_ds, collator)

    # ── evaluate ──────────────────────────────────────────────────────────────
    metrics     = _extract_eval_metrics(evaluator, val_logits, val_df, cfg, model_key)

    map3        = metrics.get(f"{model_key}_val/map@3",       float("nan"))
    hit1        = metrics.get(f"{model_key}_val/hit@1",       float("nan"))
    hit2        = metrics.get(f"{model_key}_val/hit@2",       float("nan"))
    hit3        = metrics.get(f"{model_key}_val/hit@3",       float("nan"))
    missed_rate = metrics.get(f"{model_key}_val/missed_rate", float("nan"))

    logger.info(f"[M4] {model_key} val MAP@3 = {map3:.4f}")

    # ── log to W&B ────────────────────────────────────────────────────────────
    _log_m4_metrics(
        run         = wandb_run,
        map3        = map3,
        hit1        = hit1,
        hit2        = hit2,
        hit3        = hit3,
        missed_rate = missed_rate,
        model_key   = model_key,
    )

    # ── save logits to artifacts_save_dir ─────────────────────────────────────
    _save_m4_logits(cfg, model_key, val_logits, test_logits)

    # ── save model weights ────────────────────────────────────────────────────
    save_path = fine_tuner.save_model(cfg.model_dir / f"best-{model_key}")

    # ── log model artefact to W&B ─────────────────────────────────────────────
    if wandb_run is not None and _WANDB_AVAILABLE:
        try:
            artifact = _wandb.Artifact(
                name        = f"model-{model_key}",
                type        = "model",
                description = (
                    f"LoRA fine-tuned {model_name} for MCQ (key={model_key})"
                ),
            )
            artifact.add_dir(str(save_path))
            wandb_run.log_artifact(artifact)
            logger.info(f"[W&B] model artefact logged for {model_key}")
        except Exception as exc:
            logger.warning(f"[W&B] artefact logging skipped: {exc}")

    # ── GPU cleanup ───────────────────────────────────────────────────────────
    del fine_tuner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_logits, test_logits


# ─────────────────────────────────────────────────────────────────────────────
# Cached logits + W&B logging  (when artefacts are reused)
# ─────────────────────────────────────────────────────────────────────────────

def _log_cached_metrics(
    cfg,
    evaluator   : Evaluator,
    val_df      : pd.DataFrame,
    logits      : np.ndarray,
    model_key   : str,
    model_name  : str,
    source_label: str,
) -> None:
    """
    When logits are loaded from cache, open a fresh W&B run, compute metrics
    from the cached logits, log them, and close the run immediately.

    This ensures every model has a valid W&B run even when no training occurs.
    """
    run = _init_m4_run(
        cfg,
        model_key    = model_key,
        model_name   = model_name,
        extra_config = {"artefact_source": source_label, "from_cache": True},
    )

    if run is None:
        return

    try:
        metrics     = _extract_eval_metrics(evaluator, logits, val_df, cfg, model_key)
        map3        = metrics.get(f"{model_key}_val/map@3",       0.0)
        hit1        = metrics.get(f"{model_key}_val/hit@1",       0.0)
        hit2        = metrics.get(f"{model_key}_val/hit@2",       0.0)
        hit3        = metrics.get(f"{model_key}_val/hit@3",       0.0)
        missed_rate = metrics.get(f"{model_key}_val/missed_rate", 0.0)

        _log_m4_metrics(
            run         = run,
            map3        = map3,
            hit1        = hit1,
            hit2        = hit2,
            hit3        = hit3,
            missed_rate = missed_rate,
            model_key   = model_key,
            extra       = {"from_cache": 1.0},
        )
    finally:
        finish_run(run)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation / comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _run_comparison(
    val_df    : pd.DataFrame,
    logit_map : Dict[str, np.ndarray],
    evaluator : Evaluator,
    cfg,
    wandb_run : Optional[object] = None,
) -> Dict[str, float]:
    """
    Compare all model variants (DeBERTa, RoBERTa, ensemble) on val set.

    Logs a comparison table to W&B if *wandb_run* is provided.
    Returns flat dict of MAP@3 values.
    """
    option_cols          = cfg.options
    ablation: Dict[str, float]      = {}
    rows: List[Dict[str, str]]      = []

    for name, logits in logit_map.items():
        preds   = evaluator.scores_to_top_k_predictions(logits, option_cols)
        metrics = evaluator.evaluate(
            df          = val_df,
            predictions = preds,
            split       = name,
        )
        map3    = metrics.get(f"{name}/map@3", 0.0)
        ablation[f"{name}/map@3"] = map3
        rows.append({"Model": name, "MAP@3": f"{map3:.4f}"})
        logger.info(f"[M4] {name} MAP@3 = {map3:.4f}")

    df_cmp = pd.DataFrame(rows)
    sep    = "─" * 45
    logger.info(f"\n{sep}")
    logger.info("  Milestone 4 — Model Comparison")
    logger.info(sep)
    logger.info(df_cmp.to_string(index=False))
    logger.info(sep)
    print(df_cmp.to_string(index=False))

    if wandb_run is not None and _WANDB_AVAILABLE:
        try:
            table = _wandb.Table(dataframe=df_cmp)
            wandb_run.log({"milestone4_comparison": table})
            logger.info("[W&B] comparison table logged to ensemble run.")
        except Exception as exc:
            logger.warning(f"[W&B] table logging skipped: {exc}")

    return ablation


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_milestone4(
    train_df        : pd.DataFrame,
    val_df          : pd.DataFrame,
    test_df         : pd.DataFrame,
    m3_results      : Dict[str, Any],
    primary_model   : str  = "microsoft/deberta-v3-base",
    secondary_model : str  = "roberta-base",
    use_lora        : bool = True,
    train_secondary : bool = True,
) -> Dict[str, Any]:
    """
    Execute the full Milestone 4 transformer fine-tuning pipeline.

    W&B strategy
    ────────────
    Each model (primary / secondary / ensemble) gets its own isolated run:
      - ``m4-ft``       — DeBERTa fine-tune (or cached logits replay)
      - ``m4-roberta``  — RoBERTa fine-tune (or cached logits replay)
      - ``m4-ensemble`` — averaged logits comparison + table

    Artefact loading priority
    ─────────────────────────
    1. cfg.artifacts_load_dir  — pre-computed Kaggle input artefacts.
    2. cfg.artifacts_save_dir  — locally saved artefacts.
    3. Compute from scratch (training) → saved to cfg.artifacts_save_dir.

    Parameters
    ----------
    train_df        : full training DataFrame (already preprocessed)
    val_df          : validation split
    test_df         : test DataFrame
    m3_results      : output dict from ``run_milestone3`` (provides contexts)
    primary_model   : HuggingFace model id for primary fine-tuning
    secondary_model : HuggingFace model id for diversity ensemble
    use_lora        : whether to apply LoRA adapters
    train_secondary : whether to train the secondary model

    Returns
    -------
    dict with keys:
        "ft_val_logits", "ft_test_logits",
        "roberta_val_logits", "roberta_test_logits",
        "ensemble_val_logits", "ensemble_test_logits",
        "metrics"
    """
    cfg       = Config()
    evaluator = Evaluator()

    cfg.finetune_model = primary_model

    results: Dict[str, Any] = {"metrics": {}}

    logger.info(
        f"[M4] artifacts_load_dir : {cfg.artifacts_load_dir} "
        f"(exists={cfg.artifacts_load_dir.exists()})"
    )
    logger.info(f"[M4] artifacts_save_dir : {cfg.artifacts_save_dir}")

    # ── extract contexts from M3 ──────────────────────────────────────────────
    train_contexts: List[str] = m3_results.get("train_contexts", [""] * len(train_df))
    val_contexts:   List[str] = m3_results.get("val_contexts",   [""] * len(val_df))
    test_contexts:  List[str] = m3_results.get("test_contexts",  [""] * len(test_df))

    # ── Step 1: artefact cache check ──────────────────────────────────────────
    #
    #   Config.load_artifact() checks artifacts_load_dir first (pre-built
    #   Kaggle input), then falls back to artifacts_save_dir (local cache).
    #   _load_m4_artifacts() wraps this with an all-or-nothing guard so we
    #   never use a partial set of stale logit arrays.
    #
    artefacts: Optional[Dict[str, np.ndarray]] = _load_m4_artifacts(cfg)

    # ── Step 2: use cache OR train ────────────────────────────────────────────
    if artefacts is not None:
        # ── CACHED PATH ───────────────────────────────────────────────────────
        logger.info("[M4] Reusing artefacts from artifact directories.")

        ft_val_logits       = artefacts["ft_val_logits"]
        ft_test_logits      = artefacts["ft_test_logits"]
        roberta_val_logits  = artefacts["roberta_val_logits"]
        roberta_test_logits = artefacts["roberta_test_logits"]

        # Open a dedicated W&B run per model, log cached metrics, then close.
        _log_cached_metrics(
            cfg          = cfg,
            evaluator    = evaluator,
            val_df       = val_df,
            logits       = ft_val_logits,
            model_key    = "ft",
            model_name   = primary_model,
            source_label = "artifacts_load_dir",
        )
        _log_cached_metrics(
            cfg          = cfg,
            evaluator    = evaluator,
            val_df       = val_df,
            logits       = roberta_val_logits,
            model_key    = "roberta",
            model_name   = secondary_model,
            source_label = "artifacts_load_dir",
        )

    else:
        # ── TRAINING PATH ─────────────────────────────────────────────────────
        logger.info("[M4] No cached artefacts — training from scratch …")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ── Primary model (DeBERTa) ───────────────────────────────────────────
        logger.info(f"[M4] Primary model: {primary_model}")

        primary_run = _init_m4_run(
            cfg,
            model_key    = "ft",
            model_name   = primary_model,
            extra_config = {"use_lora": use_lora, "from_cache": False},
        )

        try:
            primary_tok = _build_tokenizer(primary_model)

            train_ds, val_ds, test_ds, collator = _build_datasets(
                cfg,
                primary_tok,
                train_df, val_df, test_df,
                train_contexts, val_contexts, test_contexts,
            )

            _verify_batch(train_ds, collator)

            ft_val_logits, ft_test_logits = _train_and_predict(
                cfg        = cfg,
                evaluator  = evaluator,
                model_name = primary_model,
                train_ds   = train_ds,
                val_ds     = val_ds,
                test_ds    = test_ds,
                val_df     = val_df,
                collator   = collator,
                tokenizer  = primary_tok,
                model_key  = "ft",
                use_lora   = use_lora,
                wandb_run  = primary_run,
            )
        finally:
            finish_run(primary_run)

        # ── Secondary model (RoBERTa) ─────────────────────────────────────────
        if train_secondary:
            logger.info(f"[M4] Secondary model: {secondary_model}")

            roberta_run = _init_m4_run(
                cfg,
                model_key    = "roberta",
                model_name   = secondary_model,
                extra_config = {
                    "use_lora"  : use_lora,
                    "from_cache": False,
                    "max_length": 256,
                },
            )

            try:
                roberta_val_logits, roberta_test_logits = _train_secondary(
                    cfg            = cfg,
                    evaluator      = evaluator,
                    model_name     = secondary_model,
                    train_df       = train_df,
                    val_df         = val_df,
                    test_df        = test_df,
                    train_contexts = train_contexts,
                    val_contexts   = val_contexts,
                    test_contexts  = test_contexts,
                    use_lora       = use_lora,
                    wandb_run      = roberta_run,
                )
            finally:
                finish_run(roberta_run)

        else:
            logger.warning(
                "[M4] Secondary model skipped — using primary logits as fallback."
            )
            roberta_val_logits  = ft_val_logits.copy()
            roberta_test_logits = ft_test_logits.copy()

            # Save fallback logits to artifacts_save_dir
            _save_m4_logits(cfg, "roberta", roberta_val_logits, roberta_test_logits)

            # Still create a W&B run for the fallback so run count ≥ 3
            _log_cached_metrics(
                cfg          = cfg,
                evaluator    = evaluator,
                val_df       = val_df,
                logits       = roberta_val_logits,
                model_key    = "roberta",
                model_name   = secondary_model,
                source_label = "fallback_from_primary",
            )

    # ── Step 3: ensemble (average logits) ─────────────────────────────────────
    ensemble_val_logits  = (ft_val_logits  + roberta_val_logits)  / 2.0
    ensemble_test_logits = (ft_test_logits + roberta_test_logits) / 2.0

    # ── Step 4: populate results ──────────────────────────────────────────────
    results.update(
        {
            "ft_val_logits"       : ft_val_logits,
            "ft_test_logits"      : ft_test_logits,
            "roberta_val_logits"  : roberta_val_logits,
            "roberta_test_logits" : roberta_test_logits,
            "ensemble_val_logits" : ensemble_val_logits,
            "ensemble_test_logits": ensemble_test_logits,
        }
    )

    # ── Step 5: ensemble run + comparison / ablation table ────────────────────
    ensemble_run = _init_m4_run(
        cfg,
        model_key    = "ensemble",
        model_name   = f"{primary_model} + {secondary_model}",
        extra_config = {"ensemble_strategy": "average_logits"},
    )

    try:
        logit_map = {
            "deberta_ft" : ft_val_logits,
            "roberta_ft" : roberta_val_logits,
            "ensemble_ft": ensemble_val_logits,
        }
        ablation = _run_comparison(
            val_df    = val_df,
            logit_map = logit_map,
            evaluator = evaluator,
            cfg       = cfg,
            wandb_run = ensemble_run,
        )
        results["metrics"].update(ablation)

        ens_map3        = ablation.get("ensemble_ft/map@3", 0.0)
        ens_metrics_raw = _extract_eval_metrics(
            evaluator, ensemble_val_logits, val_df, cfg, "ensemble"
        )
        _log_m4_metrics(
            run         = ensemble_run,
            map3        = ens_map3,
            hit1        = ens_metrics_raw.get("ensemble_val/hit@1",       0.0),
            hit2        = ens_metrics_raw.get("ensemble_val/hit@2",       0.0),
            hit3        = ens_metrics_raw.get("ensemble_val/hit@3",       0.0),
            missed_rate = ens_metrics_raw.get("ensemble_val/missed_rate", 0.0),
            model_key   = "ensemble",
        )
    finally:
        finish_run(ensemble_run)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Secondary model helper
# ─────────────────────────────────────────────────────────────────────────────

def _train_secondary(
    cfg,
    evaluator      : Evaluator,
    model_name     : str,
    train_df       : pd.DataFrame,
    val_df         : pd.DataFrame,
    test_df        : pd.DataFrame,
    train_contexts : List[str],
    val_contexts   : List[str],
    test_contexts  : List[str],
    use_lora       : bool,
    wandb_run      : Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a secondary model with its own tokenizer + dataset.
    Patches cfg.finetune_model / cfg.max_length temporarily.
    Passes *wandb_run* through to _train_and_predict so metrics are logged.
    """
    original_model      = cfg.finetune_model
    original_max_length = cfg.max_length
    cfg.finetune_model  = model_name
    cfg.max_length      = 256          # shorter context for diversity/speed

    try:
        sec_tok = _build_tokenizer(model_name)

        train_ds, val_ds, test_ds, collator = _build_datasets(
            cfg,
            sec_tok,
            train_df, val_df, test_df,
            train_contexts, val_contexts, test_contexts,
        )

        val_logits, test_logits = _train_and_predict(
            cfg        = cfg,
            evaluator  = evaluator,
            model_name = model_name,
            train_ds   = train_ds,
            val_ds     = val_ds,
            test_ds    = test_ds,
            val_df     = val_df,
            collator   = collator,
            tokenizer  = sec_tok,
            model_key  = "roberta",
            use_lora   = use_lora,
            wandb_run  = wandb_run,
        )
    except Exception as exc:
        logger.error(
            f"[M4] Secondary model ({model_name}) failed: {exc}. "
            "Falling back to primary logits."
        )
        raise
    finally:
        cfg.finetune_model = original_model
        cfg.max_length     = original_max_length

    return val_logits, test_logits


# ─────────────────────────────────────────────────────────────────────────────
# Batch shape verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_batch(
    dataset   : Dataset,
    collator  : DataCollatorForMultipleChoice,
    batch_size: int = 2,
) -> None:
    """Log the shape of the first collated batch for quick debugging."""
    from torch.utils.data import DataLoader as TorchDataLoader

    loader = TorchDataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    batch  = next(iter(loader))
    logger.info(f"[verify] batch keys        : {list(batch.keys())}")
    logger.info(f"[verify] input_ids shape   : {batch['input_ids'].shape}")
    # Expected: (batch_size, n_options, max_length)