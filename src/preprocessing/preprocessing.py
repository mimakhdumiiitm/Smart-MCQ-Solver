# preprocessing.py
# preprocessing/preprocessor.py
# Text cleaning + feature-column construction.
# Saves / loads processed CSVs so the pipeline never re-does this work.

import re
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from config.config import Config
from utils.persistence import save_dataframe, load_dataframe

logger = logging.getLogger("Preprocessor")


class TextPreprocessor:
    """
    Production-grade text preprocessor for MCQ DataFrames.

    Pipeline
    --------
    1. Strip HTML / URLs / extra whitespace.
    2. Optionally lowercase.
    3. Add *_clean columns for every text column.
    4. Build combined columns used by rankers.
    5. Persist the result to CSV for reuse.

    Design note
    -----------
    Compiled regex patterns are class-level constants — compiled once,
    reused for every call to `clean_text`.
    """

    _HTML        = re.compile(r"<[^>]+>")
    _URL         = re.compile(r"https?://\S+|www\.\S+")
    _MULTI_SPACE = re.compile(r"\s+")

    def __init__(
        self,
        lowercase          : bool = True,
        remove_html        : bool = True,
        remove_urls        : bool = True,
        normalize_whitespace: bool = True,
        remove_special     : bool = False,
    ) -> None:
        self.lowercase           = lowercase
        self.remove_html         = remove_html
        self.remove_urls         = remove_urls
        self.normalize_whitespace = normalize_whitespace
        self.remove_special      = remove_special

        self._SPECIAL = re.compile(
            r"[^a-zA-Z0-9\s\.,!?;:\-\'\"()\[\]{}%$#@&*+=/\\<>]"
        )

    # ─────────────────────────────────────────
    # Single-string cleaning
    # ─────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Apply the configured cleaning steps to one string."""
        if not isinstance(text, str) or not text.strip():
            return ""
        if self.remove_html:
            text = self._HTML.sub(" ", text)
        if self.remove_urls:
            text = self._URL.sub(" ", text)
        if self.remove_special:
            text = self._SPECIAL.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        if self.normalize_whitespace:
            text = self._MULTI_SPACE.sub(" ", text).strip()
        return text

    # ─────────────────────────────────────────
    # DataFrame-level processing
    # ─────────────────────────────────────────

    def process_dataframe(
        self,
        df    : pd.DataFrame,
        config: Config,
        split : str = "train",       # "train" | "test"  — used for cache path
    ) -> pd.DataFrame:
        """
        Clean all text columns and add derived feature columns.

        Cache behaviour
        ---------------
        * If `config.use_cached_processed` is True **and** the processed CSV
          exists, it is loaded directly — no cleaning is performed.
        * Otherwise the pipeline runs and the result is saved.

        Returns
        -------
        pd.DataFrame  with *_clean and *_combined columns added.
        """
        cache_path: Path = (
            config.processed_train_path
            if split == "train"
            else config.processed_test_path
        )

        # ── Try cache ────────────────────────────────────────────
        if config.use_cached_processed:
            cached = load_dataframe(cache_path)
            if cached is not None:
                logger.info(f"[{split}] Using cached processed data: {cache_path}")
                return cached

        # ── Run pipeline ─────────────────────────────────────────
        logger.info(f"[{split}] Running preprocessing pipeline …")
        df = df.copy()
        option_cols = [c for c in config.options if c in df.columns]

        df["prompt_clean"] = df[config.prompt_col].apply(self.clean_text)

        for opt in option_cols:
            df[f"{opt}_clean"] = df[opt].apply(self.clean_text)

        # Combined: prompt + all options (used by W2V trainer)
        df["all_text"] = (
            df["prompt_clean"] + " " +
            df[[f"{o}_clean" for o in option_cols]].apply(
                lambda row: " ".join(row), axis=1
            )
        )

        # Per-option combined text (used by TF-IDF / W2V rankers)
        for opt in option_cols:
            df[f"prompt_{opt}_combined"] = (
                df["prompt_clean"] + " " + df[f"{opt}_clean"]
            )

        # ── Save ─────────────────────────────────────────────────
        save_dataframe(df, cache_path)
        logger.info(
            f"[{split}] Preprocessing complete. "
            f"New columns: {[c for c in df.columns if '_clean' in c or 'combined' in c]}"
        )
        return df

    # ─────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────

    @staticmethod
    def get_option_texts(
        df    : pd.DataFrame,
        config: Config,
    ) -> Dict[str, List[str]]:
        """Return ``{option: [text, ...]}`` for every option column."""
        return {
            opt: df[f"{opt}_clean"].fillna("").tolist()
            for opt in config.options
            if opt in df.columns
        }