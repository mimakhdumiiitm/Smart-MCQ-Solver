# src/pipelines/inference_pipeline.py

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from config.config import Config
from src.data.dataset import DataCollatorForMultipleChoice, MCQDatasetBuilder
from src.ensemble.fuser import Fuser
from src.evaluation.evaluator import Evaluator
from src.models.transformer_ranker import MCQFineTuner
from src.utils.submission import SubmissionGenerator

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Load saved fine-tuned models and generate test predictions.

    Usage:
        pipeline = InferencePipeline(cfg)
        submission = pipeline.run(test_df, test_contexts)
        # submission: pd.DataFrame with columns [id, prediction]
    """

    def __init__(self, config: Config):
        self.config = config
        self.evaluator = Evaluator(config)
        self.fuser = Fuser(config)
        self.submission_gen = SubmissionGenerator(config)

    def _load_logits_or_predict(
        self,
        model_key: str,
        model_name: str,
        test_df: pd.DataFrame,
        test_ctx: List[str],
        max_length: int,
    ) -> np.ndarray:
        """
        Try loading cached logits first; fall back to live inference.

        Cache path: config.output_dir/{model_key}_test_logits.npy
        This avoids re-running expensive inference if logits were
        already saved during training.
        """
        cache_path = self.config.output_dir / f"{model_key}_test_logits.npy"

        if cache_path.exists():
            logger.info(f"[{model_key}] Loading cached logits: {cache_path}")
            return np.load(cache_path)

        logger.info(f"[{model_key}] No cache found — running inference.")

        # Build tokenizer + dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        orig_len = self.config.max_length
        self.config.max_length = max_length
        builder = MCQDatasetBuilder(self.config, tokenizer)
        test_ds = builder.build_dataset(
            test_df, contexts=test_ctx, include_labels=False
        )
        self.config.max_length = orig_len

        collator = DataCollatorForMultipleChoice(
            tokenizer=tokenizer, padding=True, max_length=max_length
        )

        # Load model and predict
        model_path = self.config.model_dir / f"best-{model_key}"
        tuner = MCQFineTuner(self.config, self.evaluator)
        tuner.load_model(model_path, model_name=model_name)
        logits = tuner.predict(test_ds, collator)

        np.save(cache_path, logits)
        tuner.cleanup()
        return logits

    def run(
        self,
        test_df: pd.DataFrame,
        test_ctx: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate ensemble submission from all saved models.

        Steps:
          1. Load / compute logits for each model
          2. Fuse logits via weighted average (reuses fuser.py)
          3. Convert to top-3 predictions
          4. Build submission DataFrame

        Returns:
            pd.DataFrame with columns [id, prediction]
        """
        test_ctx = test_ctx or [""] * len(test_df)

        all_logits: Dict[str, np.ndarray] = {}

        # Primary model
        all_logits["deberta"] = self._load_logits_or_predict(
            model_key="deberta",
            model_name=self.config.finetune_model,
            test_df=test_df,
            test_ctx=test_ctx,
            max_length=self.config.max_length,
        )

        # Secondary model
        all_logits["roberta"] = self._load_logits_or_predict(
            model_key="roberta",
            model_name=self.config.secondary_model,
            test_df=test_df,
            test_ctx=test_ctx,
            max_length=256,
        )

        # Ensemble (reuse Milestone 3 fuser)
        fused_logits = self.fuser.fuse(all_logits)

        # Top-3 predictions
        predictions = self.evaluator.scores_to_top_k_predictions(
            fused_logits, self.config.options
        )

        # Submission DataFrame
        submission = self.submission_gen.build(test_df, predictions)
        return submission