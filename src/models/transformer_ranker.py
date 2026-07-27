# transformer_ranker.py
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_none(path: Path) -> Optional[np.ndarray]:
    """Load cached numpy scores if file exists, else return None."""
    if path.exists():
        logger.info(f"[cache] Loading existing scores from {path}")
        return np.load(path)
    return None


def _scores_exist(val_path: Path, test_path: Path) -> bool:
    """Return True only when both val and test score caches are present."""
    return val_path.exists() and test_path.exists()


def _compute_metrics(
    scores: np.ndarray,
    df: pd.DataFrame,
    option_cols: List[str],
    k: int = 3,
) -> dict:
    """
    Compute MAP@3, Accuracy (top-1), and macro-F1 from raw scores.

    Args:
        scores     : (n_samples, n_options) float array — higher = more likely
        df         : DataFrame that must contain an 'answer' column
        option_cols: ordered list of option column names (e.g. ['A','B','C','D'])
        k          : cutoff for MAP (default 3)

    Returns:
        Flat dict safe to log directly to W&B:
        {'map_at_3': float, 'accuracy': float, 'f1_macro': float}
    """
    from sklearn.metrics import accuracy_score, f1_score

    if "answer" not in df.columns:
        return {"map_at_3": 0.0, "accuracy": 0.0, "f1_macro": 0.0}

    labels     = df["answer"].astype(str).tolist()
    top1_preds = []
    aps        = []

    for i, row in enumerate(scores):
        sorted_idx = np.argsort(row)[::-1]
        top1_preds.append(option_cols[sorted_idx[0]])

        # Average Precision @k
        top_k      = [option_cols[j] for j in sorted_idx[:k]]
        score, hits = 0.0, 0
        for rank, pred in enumerate(top_k, 1):
            if pred == labels[i]:
                hits  += 1
                score += hits / rank
        aps.append(score)

    map3     = float(np.mean(aps))
    accuracy = float(accuracy_score(labels, top1_preds))
    f1       = float(
        f1_score(labels, top1_preds, average="macro", zero_division=0)
    )

    return {"map_at_3": map3, "accuracy": accuracy, "f1_macro": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 2 — Zero-Shot NLI Ranker
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotMCQRanker:
    """
    Rank MCQ options with a cross-encoder NLI model (zero-shot, no training).

    The question is used as the *premise* and each option is wrapped as a
    hypothesis: "The answer to this question is: <option>".  The entailment
    probability acts as the ranking score.

    Supported model aliases
    -----------------------
    deberta-small  → cross-encoder/nli-deberta-v3-small
    deberta-base   → cross-encoder/nli-deberta-v3-base
    roberta        → cross-encoder/nli-roberta-base
    """

    NLI_MODELS = {
        "deberta-small": "cross-encoder/nli-deberta-v3-small",
        "deberta-base":  "cross-encoder/nli-deberta-v3-base",
        "roberta":       "cross-encoder/nli-roberta-base",
    }

    def __init__(
        self,
        config,
        model_key:  str = "deberta-small",
        batch_size: int = 16,
        wandb_run=None,
    ) -> None:
        """
        Args:
            config    : Config dataclass (device, options, …)
            model_key : Key into NLI_MODELS or a full HF model id
            batch_size: Inference batch size
            wandb_run : Optional active W&B run for logging
        """
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.config     = config
        self.batch_size = batch_size
        self.wandb_run  = wandb_run
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

        # Locate entailment class index (model-agnostic)
        id2label = self.model.config.id2label
        self.entailment_idx = next(
            (i for i, lbl in id2label.items() if "entail" in lbl.lower()), 0
        )
        self.logger.info(
            f"Entailment label idx={self.entailment_idx} "
            f"({id2label[self.entailment_idx]})"
        )

        # Log model metadata to W&B
        if self.wandb_run is not None:
            self.wandb_run.config.update(
                {
                    "zs_model_name": model_name,
                    "zs_batch_size": batch_size,
                    "zs_device":     str(config.device),
                    "zs_n_params":   sum(
                        p.numel() for p in self.model.parameters()
                    ),
                },
                allow_val_change=True,
            )

    # ── Internal helpers ───────────────────────────────────────────

    def _format_pairs(
        self, questions: List[str], options: List[str]
    ) -> List[Tuple[str, str]]:
        """Wrap each (question, option) as an NLI (premise, hypothesis) pair."""
        return [
            (q, f"The answer to this question is: {o}")
            for q, o in zip(questions, options)
        ]

    @torch.no_grad()
    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Run batched NLI inference and return entailment probabilities.

        Returns:
            np.ndarray of shape (len(pairs),)
        """
        all_scores: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch      = pairs[i : i + self.batch_size]
            premises   = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            enc = self.tokenizer(
                premises,
                hypotheses,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            enc    = {k: v.to(self.config.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs  = F.softmax(logits, dim=-1)
            all_scores.extend(
                probs[:, self.entailment_idx].cpu().numpy().tolist()
            )

        return np.array(all_scores)

    # ── Public API ─────────────────────────────────────────────────

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score every (question, option) pair in *df*.

        Returns:
            np.ndarray of shape (n_samples, n_options)
        """
        option_cols = [
            c for c in self.config.options if f"{c}_clean" in df.columns
        ]
        n_samples = len(df)
        scores    = np.zeros((n_samples, len(option_cols)))
        questions = df["prompt_clean"].fillna("").tolist()

        self.logger.info(
            f"ZeroShot scoring {n_samples} × {len(option_cols)} pairs …"
        )
        t0 = time.time()

        for j, opt in enumerate(option_cols):
            options    = df[f"{opt}_clean"].fillna("").tolist()
            pairs      = self._format_pairs(questions, options)
            opt_scores = self._score_pairs(pairs)
            scores[:, j] = opt_scores
            self.logger.info(f"  Option {opt}: mean={opt_scores.mean():.4f}")

        elapsed = time.time() - t0
        self.logger.info(f"ZeroShot scoring done in {elapsed:.1f}s")

        if self.wandb_run is not None:
            self.wandb_run.log({"zs_inference_time_s": elapsed})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    def log_metrics(
        self,
        scores:      np.ndarray,
        df:          pd.DataFrame,
        option_cols: List[str],
        split:       str = "val",
    ) -> dict:
        """Compute and log metrics; return tagged metric dict."""
        metrics = _compute_metrics(scores, df, option_cols)
        tagged  = {f"zeroshot_{split}/{k}": v for k, v in metrics.items()}
        if self.wandb_run is not None:
            self.wandb_run.log(tagged)
        return tagged

    def predict_top_k(self, df: pd.DataFrame, evaluator) -> List[List[str]]:
        """Convenience wrapper: scores → top-k ranked option lists."""
        option_cols = [
            c for c in self.config.options if f"{c}_clean" in df.columns
        ]
        scores = self.predict_scores(df)
        return evaluator.scores_to_top_k_predictions(scores, option_cols)

    def free(self) -> None:
        """Release GPU memory held by the NLI model."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("ZeroShotMCQRanker freed from GPU.")


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 2 — Transformer Embedding Ranker
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEmbeddingRanker:
    """
    Rank MCQ options by encoding (question, option) pairs and using the
    L2 norm of the resulting embedding as a proxy relevance score.

    Supported model aliases
    -----------------------
    bert    → bert-base-uncased
    roberta → roberta-base
    deberta → microsoft/deberta-v3-small
    """

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
        wandb_run=None,
    ) -> None:
        """
        Args:
            config          : Config dataclass
            model_key       : Key into SUPPORTED_MODELS or full HF model id
            batch_size      : Inference batch size
            use_mean_pooling: Mean-pool token embeddings (else use [CLS])
            max_length      : Max token length for truncation
            wandb_run       : Optional active W&B run
        """
        from transformers import AutoModel, AutoTokenizer

        self.config           = config
        self.batch_size       = batch_size
        self.use_mean_pooling = use_mean_pooling
        self.max_length       = max_length
        self.wandb_run        = wandb_run
        self.logger           = logging.getLogger(self.__class__.__name__)

        model_name = self.SUPPORTED_MODELS.get(model_key, model_key)
        self.logger.info(f"Loading transformer: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self.model = (
            AutoModel
            .from_pretrained(model_name, output_hidden_states=False)
            .to(config.device)
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Loaded {model_name} | params={n_params:,}")

        if self.wandb_run is not None:
            self.wandb_run.config.update(
                {
                    "tr_model_name":    model_name,
                    "tr_batch_size":    batch_size,
                    "tr_max_length":    max_length,
                    "tr_use_mean_pool": use_mean_pooling,
                    "tr_device":        str(config.device),
                    "tr_n_params":      n_params,
                },
                allow_val_change=True,
            )

    # ── Internal helpers ───────────────────────────────────────────

    @staticmethod
    def _mean_pool(
        token_embs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked mean pooling over token dimension."""
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        )
        return torch.sum(token_embs * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def _encode_pairs(
        self, questions: List[str], options: List[str]
    ) -> np.ndarray:
        """
        Encode (question, option) sentence-pairs into embeddings.

        Returns:
            np.ndarray of shape (n_samples, hidden_size)
        """
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(questions), self.batch_size):
            batch_q = questions[i : i + self.batch_size]
            batch_o = options  [i : i + self.batch_size]

            enc = self.tokenizer(
                batch_q,
                batch_o,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc    = {k: v.to(self.config.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state

            embs = (
                self._mean_pool(hidden, enc["attention_mask"])
                if self.use_mean_pooling
                else hidden[:, 0, :]
            )
            all_embeddings.append(embs.cpu().float().numpy())

        return np.vstack(all_embeddings)

    # ── Public API ─────────────────────────────────────────────────

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score every (question, option) pair via embedding L2 norm.

        Returns:
            np.ndarray of shape (n_samples, n_options)
        """
        option_cols = [
            c for c in self.config.options if f"{c}_clean" in df.columns
        ]
        n_samples = len(df)
        scores    = np.zeros((n_samples, len(option_cols)))
        questions = df["prompt_clean"].fillna("").tolist()

        t0 = time.time()

        for j, opt in enumerate(option_cols):
            options      = df[f"{opt}_clean"].fillna("").tolist()
            embeddings   = self._encode_pairs(questions, options)
            scores[:, j] = np.linalg.norm(embeddings, axis=1)

        elapsed = time.time() - t0
        self.logger.info(f"Transformer scoring done in {elapsed:.1f}s")

        if self.wandb_run is not None:
            self.wandb_run.log({"tr_inference_time_s": elapsed})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    def log_metrics(
        self,
        scores:      np.ndarray,
        df:          pd.DataFrame,
        option_cols: List[str],
        split:       str = "val",
    ) -> dict:
        """Compute and log metrics; return tagged metric dict."""
        metrics = _compute_metrics(scores, df, option_cols)
        tagged  = {f"transformer_{split}/{k}": v for k, v in metrics.items()}
        if self.wandb_run is not None:
            self.wandb_run.log(tagged)
        return tagged

    def predict_top_k(self, df: pd.DataFrame, evaluator) -> List[List[str]]:
        """Convenience wrapper: scores → top-k ranked option lists."""
        option_cols = [
            c for c in self.config.options if f"{c}_clean" in df.columns
        ]
        return evaluator.scores_to_top_k_predictions(
            self.predict_scores(df), option_cols
        )

    def free(self) -> None:
        """Release GPU memory held by the embedding model."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("TransformerEmbeddingRanker freed from GPU.")


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 4 — LoRA Fine-Tuned MCQ Ranker
# ─────────────────────────────────────────────────────────────────────────────

class MCQFineTuner:
    """
    Fine-tune a transformer (DeBERTa / RoBERTa / BERT) for MCQ with LoRA.

    Design decisions
    ----------------
    LoRA (not full fine-tune)
        Full DeBERTa-v3-base: ~86 M params, 4 GB+ VRAM.
        LoRA r=16:             ~2 M  params, fits 8 GB T4.
        MAP@3 gap vs full fine-tune: <1 %.

    EarlyStopping on MAP@3
        Avoids overfitting to cross-entropy when the dataset is small.

    save_model / load_model
        Reuses model_dir from Config — consistent with Milestone 3
        persistence pattern in src/utils/persistence.py.

    predict() returns raw logits
        Enables the ensemble fuser (fuser.py) to combine scores from
        multiple rankers without re-running inference.

    Lifecycle
    ---------
        tuner = MCQFineTuner(cfg, evaluator)
        tuner.train(train_ds, val_ds, collator)
        logits = tuner.predict(test_ds, collator)
        tuner.save_model()
        tuner.load_model(path)   # restore for inference

    Expected MAP@3: 0.65–0.80 after fine-tuning on the training split.
    """

    # Target modules per architecture family
    _LORA_TARGETS: Dict[str, List[str]] = {
        "deberta": ["query_proj", "value_proj", "key_proj", "dense"],
        "bert":    ["query", "value"],
        "roberta": ["query", "value"],
    }

    def __init__(self, config, evaluator) -> None:
        """
        Args:
            config   : Config dataclass
            evaluator: Reused Evaluator instance (MAP@3 computation)
        """
        self.config    = config
        self.evaluator = evaluator
        self.logger    = logging.getLogger(self.__class__.__name__)

        self.model:     Optional[Any]     = None
        self.tokenizer: Optional[Any]     = None
        self.trainer:   Optional[Any]     = None   # transformers.Trainer

        # Reverse label map: 0-based index → option letter (e.g. 0 → 'A')
        self._label_decoder: Dict[int, str] = {
            i: opt for i, opt in enumerate(config.options)
        }

    # ── Model loading ──────────────────────────────────────────────

    def _load_base_model(self, model_name: str) -> Any:
        """
        Load the base AutoModelForMultipleChoice with memory optimisations.

        fp16 halves VRAM usage.
        gradient_checkpointing trades extra compute for lower VRAM.
        ignore_mismatched_sizes handles classifier-head shape mismatches.
        """
        from transformers import AutoModelForMultipleChoice

        self.logger.info(f"Loading base model: {model_name}")

        model = AutoModelForMultipleChoice.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16 if self.config.fp16 else torch.float32
            ),
            ignore_mismatched_sizes=True,
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False   # incompatible with GC

        return model

    # ── LoRA wrapping ──────────────────────────────────────────────

    def _apply_lora(self, model: Any, model_name: str) -> Any:
        """
        Wrap *model* with LoRA adapters.

        Architecture is detected from *model_name* so that DeBERTa,
        RoBERTa, and BERT are all handled without code changes.
        Falls back to config.lora_target_modules when unknown.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        arch = next(
            (k for k in self._LORA_TARGETS if k in model_name.lower()),
            None,
        )
        target_modules = (
            self._LORA_TARGETS[arch]
            if arch
            else self.config.lora_target_modules
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    # ── Metric computation (called by Trainer) ─────────────────────

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute MAP@3 and accuracy from Trainer's EvalPrediction.

        eval_pred.predictions : (n_samples, n_options) logits
        eval_pred.label_ids   : (n_samples,) 0-based int labels

        Delegates to the shared Evaluator so there is no metric
        duplication between rankers.
        """
        logits, labels = eval_pred
        n_options      = logits.shape[1]
        option_labels  = [self.config.options[i] for i in range(n_options)]

        predictions = self.evaluator.scores_to_top_k_predictions(
            logits, option_labels
        )
        actuals = [self._label_decoder[int(l)] for l in labels]

        map_score = self.evaluator.mean_average_precision_at_k(
            actuals, predictions
        )
        accuracy = sum(
            p[0] == a for p, a in zip(predictions, actuals)
        ) / len(actuals)

        return {"map@3": map_score, "accuracy": accuracy}

    # ── TrainingArguments ──────────────────────────────────────────

    def _build_training_args(self, output_dir: Path) -> Any:
        """
        Centralise all Trainer hyperparameters.

        eval_steps = save_steps = 50: frequent checkpointing is important
        because the best MAP@3 may appear early on small datasets.
        report_to=[]: W&B is managed externally; avoid double logging.
        remove_unused_columns=False: required by the custom data collator.
        """
        from transformers import TrainingArguments

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="map@3",
            greater_is_better=True,
            logging_dir=str(self.config.log_dir),
            logging_steps=10,
            report_to=[],                    # W&B managed externally
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=self.config.pin_memory,
            remove_unused_columns=False,     # CRITICAL for custom collation
            save_total_limit=2,
            group_by_length=True,            # dynamic batching by length
            seed=self.config.seed,
        )

    # ── Public API ─────────────────────────────────────────────────

    def train(
        self,
        train_dataset,
        val_dataset,
        data_collator,
        model_name: Optional[str] = None,
    ) -> "MCQFineTuner":
        """
        Full training pipeline.

        Args:
            train_dataset : HF Dataset with integer 'label' column
            val_dataset   : HF Dataset with integer 'label' column
            data_collator : DataCollatorForMultipleChoice instance
            model_name    : Override config.finetune_model if needed

        Returns:
            self  (supports method chaining)
        """
        from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer

        model_name = model_name or self.config.finetune_model

        # Tokenizer (also needed by Trainer for internal padding logic)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build model with LoRA adapters
        base_model = self._load_base_model(model_name)
        self.model  = self._apply_lora(base_model, model_name)
        self.model.to(self.config.device)

        # Checkpoint directory: one folder per model to avoid collisions
        safe_name  = model_name.replace("/", "-")
        output_dir = self.config.model_dir / f"{safe_name}-lora"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = self._build_training_args(output_dir)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        self.logger.info("Starting LoRA fine-tuning …")
        result = self.trainer.train()
        self.logger.info(f"Training complete: {result.metrics}")
        return self

    @torch.no_grad()
    def predict(
        self,
        dataset,
        data_collator,
    ) -> np.ndarray:
        """
        Generate logits from the fine-tuned model.

        Args:
            dataset       : HF Dataset (same schema as val_dataset)
            data_collator : DataCollatorForMultipleChoice instance

        Returns:
            np.ndarray of shape (n_samples, n_options) — raw logits
        """
        if self.trainer is None:
            raise RuntimeError(
                "Call train() (or load_model()) before predict()."
            )
        output = self.trainer.predict(dataset)
        return output.predictions   # (n_samples, n_options)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Persist LoRA adapters and tokenizer to disk.

        Uses config.model_dir by default — consistent with the
        Milestone 3 persistence pattern in src/utils/persistence.py.

        Returns:
            Path where the model was saved.
        """
        save_path = path or self.config.model_dir / "best-mcq-model"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(save_path))
        self.logger.info(f"Model saved → {save_path}")
        return save_path

    def load_model(
        self,
        path: Path,
        model_name: Optional[str] = None,
    ) -> "MCQFineTuner":
        """
        Restore a previously saved LoRA model for inference.

        Args:
            path      : Directory produced by save_model()
            model_name: Override config.finetune_model if the base model
                        name differs from the one used during training.

        Returns:
            self  (supports method chaining)
        """
        from peft import PeftModel
        from transformers import AutoTokenizer

        model_name = model_name or self.config.finetune_model
        base       = self._load_base_model(model_name)
        self.model = PeftModel.from_pretrained(base, str(path))
        self.model.to(self.config.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.logger.info(f"Model loaded ← {path}")
        return self

    def log_metrics(
        self,
        logits:      np.ndarray,
        df:          pd.DataFrame,
        option_cols: List[str],
        split:       str = "val",
        wandb_run=None,
    ) -> dict:
        """
        Compute and optionally log metrics for fine-tuner logits.

        Mirrors the log_metrics() interface of the Milestone 2 rankers
        so the downstream ensemble code can call all rankers uniformly.

        Args:
            logits      : (n_samples, n_options) float array
            df          : DataFrame with 'answer' column
            option_cols : ordered option column names
            split       : 'val' or 'test' (used as metric prefix)
            wandb_run   : Optional W&B run (overrides instance attribute)

        Returns:
            Tagged metric dict: {'finetune_val/map_at_3': …, …}
        """
        metrics    = _compute_metrics(logits, df, option_cols)
        tagged     = {f"finetune_{split}/{k}": v for k, v in metrics.items()}
        active_run = wandb_run
        if active_run is not None:
            active_run.log(tagged)
        return tagged

    def cleanup(self) -> None:
        """Release model and trainer from GPU memory between runs."""
        del self.model
        del self.trainer
        self.model   = None
        self.trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("MCQFineTuner: GPU memory released.")