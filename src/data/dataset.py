# src/data/dataset.py
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class MCQDatasetBuilder:
    def __init__(self, config, tokenizer: PreTrainedTokenizerBase):
        """
        Args:
            config  : Config dataclass (from config/config.py)
            tokenizer: HuggingFace tokenizer matched to the model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)

        # Map option letter → int index  e.g. {"A":0, "B":1, ...}
        self.label_encoder: Dict[str, int] = {
            opt: i for i, opt in enumerate(config.options)
        }

    # ── Prompt formatting ──────────────────────────────────────────
    def format_prompt_with_context(
        self,
        question: str,
        context: str = "",
        max_context_chars: int = 256,
    ) -> str:
        """
        Prepend RAG context to question (truncated to max_context_chars).

        Why truncate here (not at tokenizer level):
            Guarantees context never crowds out the question/option text
            in the final token budget.
        """
        if context and context.strip():
            ctx = context.strip()[:max_context_chars]
            return f"Context: {ctx} Question: {question}"
        return f"Question: {question}"

    # ── Single-sample tokenization ─────────────────────────────────
    def tokenize_sample(
        self,
        question: str,
        options: List[str],
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Tokenize one MCQ sample.

        AutoModelForMultipleChoice expects shape (n_choices, seq_len).
        We achieve this by repeating the question for each option and
        pairing it with the corresponding option text.

        Returns:
            {
                "input_ids"     : List[List[int]]  shape (n_options, seq_len)
                "attention_mask": List[List[int]]  shape (n_options, seq_len)
            }
        """
        formatted_q = self.format_prompt_with_context(question, context)

        # Tokenizer pairs (question_i, option_i) for each option
        encodings = self.tokenizer(
            [formatted_q] * len(options),   # same question, once per option
            options,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",           # uniform shape before collation
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    # ── Full dataset builder ───────────────────────────────────────
    def build_dataset(
        self,
        df: pd.DataFrame,
        contexts: Optional[List[str]] = None,
        include_labels: bool = True,
    ) -> Dataset:
        """
        Convert a MCQ DataFrame into a HuggingFace Dataset.

        Args:
            df            : DataFrame with columns [prompt, A, B, C, D, E, answer]
            contexts      : Optional list of RAG context strings (one per row)
            include_labels: Set False for test data (no ground-truth answer)

        Returns:
            datasets.Dataset ready for Trainer or DataLoader
        """
        # Only keep option columns that actually exist in the DataFrame
        option_cols = [c for c in self.config.options if c in df.columns]

        # Default to empty contexts when RAG is not available
        if contexts is None:
            contexts = [""] * len(df)

        samples: List[Dict[str, Any]] = []

        for idx, (i, row) in enumerate(df.iterrows()):
            question = str(row.get(self.config.prompt_col, ""))

            # Safe context lookup — works for both RangeIndex and custom index
            ctx = contexts[idx] if idx < len(contexts) else ""

            options = [str(row.get(opt, "")) for opt in option_cols]
            encoding = self.tokenize_sample(question, options, ctx)

            sample: Dict[str, Any] = {
                "input_ids":      encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "id":             str(row.get(self.config.id_col, i)),
            }

            # Attach ground-truth label (train / val only)
            if include_labels and self.config.answer_col in row.index:
                answer = row[self.config.answer_col]
                sample["labels"] = self.label_encoder.get(answer, 0)

            samples.append(sample)

        dataset = Dataset.from_list(samples)
        self.logger.info(
            f"Dataset built: {len(dataset)} samples | "
            f"features: {list(dataset.features.keys())}"
        )
        return dataset


# ── Custom Data Collator ───────────────────────────────────────────
@dataclass
class DataCollatorForMultipleChoice:
    """
    Batch collator for AutoModelForMultipleChoice.

    Problem it solves:
        Standard DataCollatorWithPadding assumes shape (seq_len,) per sample.
        For MCQ, each sample has shape (n_options, seq_len), so we must:
          1. Flatten to (batch * n_options, seq_len)
          2. Pad the flattened batch (dynamic padding → less memory)
          3. Reshape back to (batch, n_options, seq_len)

    Usage:
        collator = DataCollatorForMultipleChoice(tokenizer=tok)
        loader = DataLoader(ds, batch_size=4, collate_fn=collator)
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of dicts from Dataset.__getitem__

        Returns:
            Batched tensors with shapes:
                input_ids      : (batch, n_options, seq_len)
                attention_mask : (batch, n_options, seq_len)
                labels         : (batch,)   — only if present
        """
        batch_size = len(features)
        n_options = len(features[0]["input_ids"])

        # Extract and remove labels / ids before padding
        labels = None
        if "labels" in features[0]:
            labels = [int(f.pop("labels")) for f in features]

        # Remove non-tensor metadata
        for f in features:
            f.pop("id", None)

        # Step 1 — Flatten: (batch * n_options) individual encodings
        flat: List[Dict[str, List[int]]] = []
        for feature in features:
            for opt_idx in range(n_options):
                flat.append(
                    {
                        "input_ids":      feature["input_ids"][opt_idx],
                        "attention_mask": feature["attention_mask"][opt_idx],
                    }
                )

        # Step 2 — Dynamic padding across the flattened batch
        padded = self.tokenizer.pad(
            flat,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Step 3 — Reshape back to (batch, n_options, seq_len)
        batch: Dict[str, torch.Tensor] = {
            k: v.view(batch_size, n_options, -1)
            for k, v in padded.items()
        }

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch