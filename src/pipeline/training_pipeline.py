# src/pipelines/training_pipeline.py
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from config.config import Config
from src.data.dataset import DataCollatorForMultipleChoice, MCQDatasetBuilder
from src.evaluation.evaluator import MAPAtKEvaluator as Evaluator
from src.ensemble.fuser import Fuser
from src.models.transformer_ranker import MCQFineTuner

# Existing Milestone 3 imports (kept as-is)
from src.models.tfidf_ranker import TFIDFRanker
from src.models.w2v_ranker import W2VRanker
from src.models.sbert_ranker import SBERTRanker

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Unified training pipeline for baselines + transformers.

    Usage:
        pipeline = TrainingPipeline(cfg)
        results  = pipeline.run(train_df, val_df, test_df,
                                train_ctx, val_ctx, test_ctx)
        # results["all_val_logits"], results["all_test_logits"] → ensemble
    """

    def __init__(self, config: Config):
        self.config = config
        self.evaluator = Evaluator(config)
        self.fuser = Fuser(config)
        self.option_cols = config.options

    # ── Dataset helpers (reuse MCQDatasetBuilder) ─────────────────
    def _build_hf_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_ctx: List[str],
        val_ctx: List[str],
        test_ctx: List[str],
        model_name: str,
        max_length: int,
    ) -> Tuple[Dataset, Dataset, Dataset,
               DataCollatorForMultipleChoice]:
        """
        Build HF datasets + collator for a given tokenizer.

        Factored out so both primary and secondary model training
        call the same builder without duplication.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Temporarily override max_length for this model
        orig_len = self.config.max_length
        self.config.max_length = max_length

        builder = MCQDatasetBuilder(self.config, tokenizer)

        train_ds = builder.build_dataset(
            train_df, contexts=train_ctx, include_labels=True
        )
        val_ds = builder.build_dataset(
            val_df, contexts=val_ctx, include_labels=True
        )
        test_ds = builder.build_dataset(
            test_df, contexts=test_ctx, include_labels=False
        )

        self.config.max_length = orig_len   # restore

        collator = DataCollatorForMultipleChoice(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
        )

        return train_ds, val_ds, test_ds, collator

    # ── Transformer stage ──────────────────────────────────────────
    def _run_transformer_stage(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_ctx: List[str],
        val_ctx: List[str],
        test_ctx: List[str],
        max_length: int = 512,
        model_key: str = "primary",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fine-tune one transformer model and return (val_logits, test_logits).

        Called for both primary (DeBERTa) and secondary (RoBERTa) models.
        Identical interface ensures the ensemble stage receives consistent
        numpy arrays regardless of which architecture was used.

        Args:
            model_name : HuggingFace model identifier
            *_df       : Split DataFrames
            *_ctx      : RAG contexts for each split
            max_length : Token budget (shorter = faster for secondary)
            model_key  : Label used for logging and checkpoint naming

        Returns:
            val_logits  : (n_val,  n_options)
            test_logits : (n_test, n_options)
        """
        logger.info(f"[{model_key}] Starting fine-tuning: {model_name}")

        # Clear GPU before each training run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build HF datasets
        train_ds, val_ds, test_ds, collator = self._build_hf_datasets(
            train_df, val_df, test_df,
            train_ctx, val_ctx, test_ctx,
            model_name, max_length,
        )

        # Train
        tuner = MCQFineTuner(self.config, self.evaluator)
        tuner.train(
            train_ds, val_ds, collator,
            model_name=model_name,
        )

        # Predict
        val_logits  = tuner.predict(val_ds,  collator)
        test_logits = tuner.predict(test_ds, collator)

        # Evaluate val split (reuse existing evaluator)
        val_preds = self.evaluator.scores_to_top_k_predictions(
            val_logits, self.option_cols
        )
        metrics = self.evaluator.evaluate(
            val_df, val_preds, self.config, split=f"{model_key}_val"
        )
        logger.info(
            f"[{model_key}] MAP@3 = "
            f"{metrics.get(f'{model_key}_val/map@3', 0):.4f}"
        )

        # Persist logits + model
        np.save(
            self.config.output_dir / f"{model_key}_val_logits.npy",
            val_logits,
        )
        np.save(
            self.config.output_dir / f"{model_key}_test_logits.npy",
            test_logits,
        )
        tuner.save_model(
            self.config.model_dir / f"best-{model_key}"
        )

        # Free GPU memory before next run
        tuner.cleanup()

        return val_logits, test_logits

    # ── Master run method ──────────────────────────────────────────
    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_ctx: Optional[List[str]] = None,
        val_ctx: Optional[List[str]] = None,
        test_ctx: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute full pipeline: baselines → transformers → ensemble.

        Returns dict of all val/test logits for downstream ensemble use.
        """
        train_ctx = train_ctx or [""] * len(train_df)
        val_ctx   = val_ctx   or [""] * len(val_df)
        test_ctx  = test_ctx  or [""] * len(test_df)

        all_val_logits:  Dict[str, np.ndarray] = {}
        all_test_logits: Dict[str, np.ndarray] = {}

        # ── Stage 1: Baselines (Milestone 3 — unchanged) ──────────
        logger.info("=== Stage 1: Baseline models ===")
        for ranker_cls, key in [
            (TFIDFRanker,  "tfidf"),
            (W2VRanker,    "w2v"),
            (SBERTRanker,  "sbert"),
        ]:
            try:
                ranker = ranker_cls(self.config)
                ranker.fit(train_df)
                v_log  = ranker.predict_logits(val_df)
                t_log  = ranker.predict_logits(test_df)
                all_val_logits[key]  = v_log
                all_test_logits[key] = t_log
                logger.info(f"[{key}] baseline logits computed.")
            except Exception as exc:
                logger.warning(f"[{key}] baseline failed: {exc}")

        # ── Stage 2: Primary transformer (DeBERTa) ────────────────
        logger.info("=== Stage 2: Primary transformer (DeBERTa) ===")
        try:
            v_log, t_log = self._run_transformer_stage(
                model_name=self.config.finetune_model,
                train_df=train_df, val_df=val_df, test_df=test_df,
                train_ctx=train_ctx, val_ctx=val_ctx, test_ctx=test_ctx,
                max_length=self.config.max_length,
                model_key="deberta",
            )
            all_val_logits["deberta"]  = v_log
            all_test_logits["deberta"] = t_log
        except Exception as exc:
            logger.error(f"Primary transformer failed: {exc}")

        # ── Stage 3: Secondary transformer (RoBERTa) ──────────────
        logger.info("=== Stage 3: Secondary transformer (RoBERTa) ===")
        try:
            v_log, t_log = self._run_transformer_stage(
                model_name=self.config.secondary_model,
                train_df=train_df, val_df=val_df, test_df=test_df,
                train_ctx=train_ctx, val_ctx=val_ctx, test_ctx=test_ctx,
                max_length=256,         # shorter budget for speed
                model_key="roberta",
            )
            all_val_logits["roberta"]  = v_log
            all_test_logits["roberta"] = t_log
        except Exception as exc:
            logger.warning(
                f"Secondary transformer failed, "
                f"falling back to DeBERTa logits: {exc}"
            )
            # Fallback: reuse primary logits (no crash)
            if "deberta" in all_val_logits:
                all_val_logits["roberta"]  = all_val_logits["deberta"].copy()
                all_test_logits["roberta"] = all_test_logits["deberta"].copy()

        return {
            "all_val_logits":  all_val_logits,
            "all_test_logits": all_test_logits,
        }