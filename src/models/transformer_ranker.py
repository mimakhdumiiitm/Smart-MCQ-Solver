# transformer_ranker.py  (no core logic changed – only helper additions at bottom)
"""
transformer_ranker.py
=====================
Milestone 2: Transformer-based MCQ ranking methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_none(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        logger.info(f"[cache] Loading existing scores from {path}")
        return np.load(path)
    return None


def _scores_exist(val_path: Path, test_path: Path) -> bool:
    return val_path.exists() and test_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Zero-Shot NLI Ranker
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotMCQRanker:
    NLI_MODELS = {
        "deberta-small": "cross-encoder/nli-deberta-v3-small",
        "deberta-base":  "cross-encoder/nli-deberta-v3-base",
        "roberta":       "cross-encoder/nli-roberta-base",
    }

    def __init__(self, config, model_key: str = "deberta-small", batch_size: int = 16) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config     = config
        self.batch_size = batch_size
        self.logger     = logging.getLogger(self.__class__.__name__)

        model_name = self.NLI_MODELS.get(model_key, model_key)
        self.logger.info(f"Loading NLI model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
            .to(config.device)
        )
        self.model.eval()

        id2label = self.model.config.id2label
        self.entailment_idx = next(
            (i for i, lbl in id2label.items() if "entail" in lbl.lower()), 0
        )
        self.logger.info(
            f"Entailment label idx={self.entailment_idx} ({id2label[self.entailment_idx]})"
        )

    def _format_pairs(self, questions: List[str], options: List[str]) -> List[Tuple[str, str]]:
        return [
            (q, f"The answer to this question is: {o}")
            for q, o in zip(questions, options)
        ]

    @torch.no_grad()
    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        all_scores: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch      = pairs[i : i + self.batch_size]
            premises   = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            enc = self.tokenizer(
                premises, hypotheses,
                truncation=True, max_length=512,
                padding=True, return_tensors="pt",
            )
            enc    = {k: v.to(self.config.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs  = F.softmax(logits, dim=-1)
            all_scores.extend(probs[:, self.entailment_idx].cpu().numpy().tolist())

        return np.array(all_scores)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        option_cols = [c for c in self.config.options if f"{c}_clean" in df.columns]
        n_samples   = len(df)
        scores      = np.zeros((n_samples, len(option_cols)))
        questions   = df["prompt_clean"].fillna("").tolist()

        self.logger.info(f"ZeroShot scoring {n_samples} × {len(option_cols)} pairs …")

        for j, opt in enumerate(option_cols):
            options    = df[f"{opt}_clean"].fillna("").tolist()
            pairs      = self._format_pairs(questions, options)
            opt_scores = self._score_pairs(pairs)
            scores[:, j] = opt_scores
            self.logger.info(f"  Option {opt}: mean={opt_scores.mean():.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    def predict_top_k(self, df: pd.DataFrame, evaluator) -> List[List[str]]:
        option_cols = [c for c in self.config.options if f"{c}_clean" in df.columns]
        scores      = self.predict_scores(df)
        return evaluator.scores_to_top_k_predictions(scores, option_cols)

    def free(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("ZeroShotMCQRanker freed from GPU.")


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Embedding Ranker
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEmbeddingRanker:
    SUPPORTED_MODELS = {
        "bert":    "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-small",
    }

    def __init__(
        self,
        config,
        model_key:        str  = "deberta",
        batch_size:       int  = 16,
        use_mean_pooling: bool = True,
        max_length:       int  = 256,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.config           = config
        self.batch_size       = batch_size
        self.use_mean_pooling = use_mean_pooling
        self.max_length       = max_length
        self.logger           = logging.getLogger(self.__class__.__name__)

        model_name = self.SUPPORTED_MODELS.get(model_key, model_key)
        self.logger.info(f"Loading transformer: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = (
            AutoModel.from_pretrained(model_name, output_hidden_states=False)
            .to(config.device)
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Loaded {model_name} | params={n_params:,}")

    @staticmethod
    def _mean_pool(token_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        return torch.sum(token_embs * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def _encode_pairs(self, questions: List[str], options: List[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(questions), self.batch_size):
            batch_q = questions[i : i + self.batch_size]
            batch_o = options  [i : i + self.batch_size]

            enc = self.tokenizer(
                batch_q, batch_o,
                max_length=self.max_length,
                truncation=True, padding=True,
                return_tensors="pt",
            )
            enc    = {k: v.to(self.config.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state

            if self.use_mean_pooling:
                embs = self._mean_pool(hidden, enc["attention_mask"])
            else:
                embs = hidden[:, 0, :]

            all_embeddings.append(embs.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        option_cols = [c for c in self.config.options if f"{c}_clean" in df.columns]
        n_samples   = len(df)
        scores      = np.zeros((n_samples, len(option_cols)))
        questions   = df["prompt_clean"].fillna("").tolist()

        for j, opt in enumerate(option_cols):
            options      = df[f"{opt}_clean"].fillna("").tolist()
            embeddings   = self._encode_pairs(questions, options)
            scores[:, j] = np.linalg.norm(embeddings, axis=1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    def predict_top_k(self, df: pd.DataFrame, evaluator) -> List[List[str]]:
        option_cols = [c for c in self.config.options if f"{c}_clean" in df.columns]
        return evaluator.scores_to_top_k_predictions(self.predict_scores(df), option_cols)

    def free(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("TransformerEmbeddingRanker freed from GPU.")