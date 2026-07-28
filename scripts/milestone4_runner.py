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

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Artefact helpers  (same pattern as milestone3_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

#: All logit artefacts produced / consumed by M4.
_M4_ARTEFACTS: List[str] = [
    "ft_val_logits.npy",
    "ft_test_logits.npy",
    "roberta_val_logits.npy",
    "roberta_test_logits.npy",
]


def _try_load_artefacts(directory: Optional[Path]) -> Optional[Dict[str, np.ndarray]]:
    """
    Attempt to load all M4 logit artefacts from *directory*.

    Returns
    -------
    dict  – keyed by stem (no extension) when ALL files exist.
    None  – if directory missing or any file absent.
    """
    if directory is None:
        return None

    directory = Path(directory)
    if not directory.exists():
        logger.debug(f"[artefact] directory not found: {directory}")
        return None

    loaded: Dict[str, np.ndarray] = {}
    for name in _M4_ARTEFACTS:
        p = directory / name
        if not p.exists():
            logger.debug(f"[artefact] missing: {p}")
            return None
        loaded[name.replace(".npy", "")] = np.load(p)
        logger.info(f"[artefact] loaded {name} ← {directory}")

    return loaded


def _all_cached(directory: Path) -> bool:
    return all((directory / n).exists() for n in _M4_ARTEFACTS)


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
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_contexts: List[str],
    val_contexts: List[str],
    test_contexts: List[str],
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
        tokenizer=tokenizer,
        padding=True,
        max_length=cfg.max_length,
    )

    logger.info(
        f"Datasets ready → train:{len(train_ds)}  "
        f"val:{len(val_ds)}  test:{len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, collator


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
    use_lora=True,
)-> Tuple[np.ndarray, np.ndarray]:
    """
    Fine-tune one model and return (val_logits, test_logits).

    Persists logits + model weights to cfg.output_dir / cfg.model_dir.
    """
    fine_tuner = MCQFineTuner(cfg, evaluator)
    fine_tuner.train(train_ds, val_ds, collator, tokenizer, use_lora=use_lora)

    val_logits  = fine_tuner.predict(val_ds,  collator)
    test_logits = fine_tuner.predict(test_ds, collator)

    # ── evaluate ──────────────────────────────────────────────────────────────
    option_cols = [c for c in cfg.options if c in val_ds.features]
    # option_cols from cfg is sufficient — val_ds doesn't store DataFrame cols
    option_cols = cfg.options

    val_preds = evaluator.scores_to_top_k_predictions(val_logits, option_cols)
    metrics = evaluator.evaluate(
        df=val_df,
        predictions=val_preds,
        split=f"{model_key}_val",
    )
    map3 = metrics.get(f"{model_key}_val/map@3", float("nan"))
    logger.info(f"[M4] {model_key} val MAP@3 = {map3:.4f}")

    # ── save logits ───────────────────────────────────────────────────────────
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{model_key}_val_logits.npy",  val_logits)
    np.save(out / f"{model_key}_test_logits.npy", test_logits)

    # ── save model ────────────────────────────────────────────────────────────
    fine_tuner.save_model(cfg.model_dir / f"best-{model_key}")

    # ── GPU cleanup ───────────────────────────────────────────────────────────
    del fine_tuner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_logits, test_logits


# ─────────────────────────────────────────────────────────────────────────────
# Ablation / comparison table
# ─────────────────────────────────────────────────────────────────────────────


def _run_comparison(
    val_df: pd.DataFrame,
    logit_map: Dict[str, np.ndarray],
    evaluator: Evaluator,
    cfg,
) -> Dict[str, float]:
    """
    Compare all model variants (DeBERTa, RoBERTa, ensemble) on val set.

    Returns flat dict of MAP@3 values.
    """
    option_cols   = cfg.options
    ablation: Dict[str, float] = {}
    rows: List[Dict[str, str]] = []

    for name, logits in logit_map.items():
        preds   = evaluator.scores_to_top_k_predictions(logits, option_cols)
        metrics = evaluator.evaluate(
            df=val_df,
            predictions=preds,
            split=name,
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

    return ablation


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_milestone4(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    m3_results: Dict[str, Any],
    primary_model: str   = "microsoft/deberta-v3-base",
    secondary_model: str = "roberta-base",
    use_lora: bool       = True,
    train_secondary: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full Milestone 4 transformer fine-tuning pipeline.

    Artefact loading priority
    -------------------------
    1. ``cfg.kaggle_artifacts_dir``  — pre-computed Kaggle artefacts.
    2. ``cfg.output_dir``            — local run cache.
    3. Compute from scratch.

    Parameters
    ----------
    train_df         : full training DataFrame (already preprocessed)
    val_df           : validation split
    test_df          : test DataFrame
    m3_results       : output dict from ``run_milestone3`` (provides contexts)
    primary_model    : HuggingFace model id for primary fine-tuning
    secondary_model  : HuggingFace model id for diversity ensemble
    use_lora         : whether to apply LoRA adapters
    train_secondary  : whether to train the secondary model

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

    # Patch model name into cfg so MCQFineTuner reads the right checkpoint
    cfg.finetune_model = primary_model

    results: Dict[str, Any] = {"metrics": {}}
    out = cfg.output_dir

    # ── extract contexts from M3 ──────────────────────────────────────────────
    train_contexts: List[str] = m3_results.get("train_contexts", [""] * len(train_df))
    val_contexts:   List[str] = m3_results.get("val_contexts",   [""] * len(val_df))
    test_contexts:  List[str] = m3_results.get("test_contexts",  [""] * len(test_df))

    # ── Step 1: artefact cache check ──────────────────────────────────────────
    artefacts: Optional[Dict[str, np.ndarray]] = None
    artefact_source = "compute"

    kaggle_dir: Optional[Path] = getattr(cfg, "kaggle_artifacts_dir", None)
    if kaggle_dir is not None:
        artefacts = _try_load_artefacts(kaggle_dir)
        if artefacts:
            artefact_source = f"kaggle_artifacts ({kaggle_dir})"

    if artefacts is None and _all_cached(out):
        artefacts = _try_load_artefacts(out)
        if artefacts:
            artefact_source = f"local_cache ({out})"

    # ── Step 2: use cache OR train ────────────────────────────────────────────
    if artefacts is not None:
        logger.info(f"[M4] Reusing artefacts ← {artefact_source}")
        ft_val_logits       = artefacts["ft_val_logits"]
        ft_test_logits      = artefacts["ft_test_logits"]
        roberta_val_logits  = artefacts["roberta_val_logits"]
        roberta_test_logits = artefacts["roberta_test_logits"]

    else:
        logger.info("[M4] No cached artefacts — training from scratch …")

        # ── GPU state reset ───────────────────────────────────────────────────
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ── Primary model (DeBERTa) ───────────────────────────────────────────
        logger.info(f"[M4] Primary model: {primary_model}")
        primary_tok = _build_tokenizer(primary_model)

        train_ds, val_ds, test_ds, collator = _build_datasets(
            cfg,
            primary_tok,
            train_df, val_df, test_df,
            train_contexts, val_contexts, test_contexts,
        )

        # Sanity-check batch shape
        _verify_batch(train_ds, collator)

        ft_val_logits, ft_test_logits = _train_and_predict(
            cfg=cfg,
            evaluator=evaluator,
            model_name=primary_model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            val_df=val_df,
            collator=collator,
            tokenizer=primary_tok,
            model_key="ft",
            use_lora=use_lora,
        )

        # ── Secondary model (RoBERTa) ─────────────────────────────────────────
        if train_secondary:
            logger.info(f"[M4] Secondary model: {secondary_model}")
            roberta_val_logits, roberta_test_logits = _train_secondary(
                cfg             = cfg,
                evaluator       = evaluator,
                model_name      = secondary_model,
                train_df        = train_df,
                val_df          = val_df,
                test_df         = test_df,
                train_contexts  = train_contexts,
                val_contexts    = val_contexts,
                test_contexts   = test_contexts,
                use_lora        = use_lora,
            )
        else:
            logger.warning(
                "[M4] Secondary model skipped — using primary logits as fallback."
            )
            roberta_val_logits  = ft_val_logits.copy()
            roberta_test_logits = ft_test_logits.copy()
            np.save(out / "roberta_val_logits.npy",  roberta_val_logits)
            np.save(out / "roberta_test_logits.npy", roberta_test_logits)

    # ── Step 3: ensemble (average logits) ────────────────────────────────────
    ensemble_val_logits  = (ft_val_logits  + roberta_val_logits)  / 2.0
    ensemble_test_logits = (ft_test_logits + roberta_test_logits) / 2.0

    # ── Step 4: populate results ──────────────────────────────────────────────
    results.update(
        {
            "ft_val_logits":        ft_val_logits,
            "ft_test_logits":       ft_test_logits,
            "roberta_val_logits":   roberta_val_logits,
            "roberta_test_logits":  roberta_test_logits,
            "ensemble_val_logits":  ensemble_val_logits,
            "ensemble_test_logits": ensemble_test_logits,
        }
    )

    # ── Step 5: comparison / ablation table ───────────────────────────────────
    logit_map = {
        "deberta_ft":  ft_val_logits,
        "roberta_ft":  roberta_val_logits,
        "ensemble_ft": ensemble_val_logits,
    }
    results["metrics"].update(
        _run_comparison(val_df, logit_map, evaluator, cfg)
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Secondary model helper (keeps run_milestone4 readable)
# ─────────────────────────────────────────────────────────────────────────────


def _train_secondary(
    cfg,
    evaluator: Evaluator,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_contexts: List[str],
    val_contexts: List[str],
    test_contexts: List[str],
    use_lora: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a secondary model with its own tokenizer + dataset.
    Patches cfg.finetune_model temporarily.
    """
    original_model = cfg.finetune_model
    cfg.finetune_model = model_name
    # Shorter sequence for speed — secondary model trades length for diversity
    original_max_length = cfg.max_length
    cfg.max_length      = 256

    try:
        sec_tok = _build_tokenizer(model_name)

        train_ds, val_ds, test_ds, collator = _build_datasets(
            cfg,
            sec_tok,
            train_df, val_df, test_df,
            train_contexts, val_contexts, test_contexts,
        )

        val_logits, test_logits = _train_and_predict(
            cfg=cfg,
            evaluator=evaluator,
            model_name=model_name,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            val_df=val_df,
            collator=collator,
            tokenizer=sec_tok,
            model_key="roberta",
            use_lora=use_lora,
        )
    except Exception as exc:
        logger.error(
            f"[M4] Secondary model ({model_name}) failed: {exc}. "
            "Falling back to primary logits."
        )
        # safe fallback — caller handles None check
        raise
    finally:
        # Always restore cfg
        cfg.finetune_model = original_model
        cfg.max_length     = original_max_length

    return val_logits, test_logits


# ─────────────────────────────────────────────────────────────────────────────
# Batch shape verification
# ─────────────────────────────────────────────────────────────────────────────


def _verify_batch(
    dataset: Dataset,
    collator: DataCollatorForMultipleChoice,
    batch_size: int = 2,
) -> None:
    """Log the shape of the first collated batch for quick debugging."""
    from torch.utils.data import DataLoader as TorchDataLoader

    loader = TorchDataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    batch  = next(iter(loader))
    logger.info(f"[verify] batch keys        : {list(batch.keys())}")
    logger.info(f"[verify] input_ids shape   : {batch['input_ids'].shape}")
    # Expected: (batch_size, n_options, max_length)