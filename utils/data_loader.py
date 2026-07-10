# utils/data_loader.py
# updated for milestone 2

import os
import re
import pandas as pd
from typing import List

from config.config import (
    TRAIN_PATH, TEST_PATH, SUBMISSION_PATH,
    TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH,
    ID_COL, ANSWER_COL, OPTION_COLS, TEXT_COLS,
    TRAIN_OUTPUT_PATH,TEST_OUTPUT_PATH
)

# ==================================================================
# EXISTING FUNCTIONS — UNCHANGED
# ==================================================================
def load_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Make sure the file exists at the specified path."
        )
    df = pd.read_csv(filepath)
    print(f"Loaded  : {filepath}")
    print(f"Shape   : {df.shape}")
    return df


def load_train(path: str = TRAIN_PATH) -> pd.DataFrame:
    """Load training data."""
    return load_csv(path)


def load_test(path: str = TEST_PATH) -> pd.DataFrame:
    """Load test data."""
    return load_csv(path)


def load_submission(path: str = SUBMISSION_PATH) -> pd.DataFrame:
    """Load sample submission file."""
    return load_csv(path)


def validate_dataframe(df: pd.DataFrame,
                        required_cols: list,
                        name: str = "DataFrame") -> bool:
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{name} is missing columns: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    print(f"{name} validation passed. All required columns present.")
    return True


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct"  : missing_pct
    })
    report = report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    return report


def fill_missing_text(df: pd.DataFrame,
                       text_cols: list = None,
                       fill_value: str = "") -> pd.DataFrame:

    if text_cols is None:
        text_cols = TEXT_COLS

    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(fill_value)
                print(f"Filled {n_missing} missing values in column '{col}'")
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "n_rows"        : len(df),
        "n_cols"        : len(df.columns),
        "columns"       : df.columns.tolist(),
        "dtypes"        : df.dtypes.to_dict(),
        "memory_kb"     : df.memory_usage(deep=True).sum() / 1024,
        "total_missing" : df.isnull().sum().sum(),
    }
    if ANSWER_COL in df.columns:
        stats["answer_distribution"] = df[ANSWER_COL].value_counts().to_dict()
    return stats


def print_basic_stats(df: pd.DataFrame, name: str = "Dataset") -> None:
    stats = basic_stats(df)
    separator = "-" * 40
    print(separator)
    print(f"BASIC STATS : {name}")
    print(separator)
    print(f"Rows        : {stats['n_rows']:,}")
    print(f"Columns     : {stats['n_cols']}")
    print(f"Memory      : {stats['memory_kb']:.2f} KB")
    print(f"Missing     : {stats['total_missing']}")
    if "answer_distribution" in stats:
        print(f"Answer Dist : {stats['answer_distribution']}")
    print(separator)


# ==================================================================
# NEW: TRANSFORMER PIPELINE DATA LOADER CLASS
# ==================================================================
class TransformerDataLoader:
    def __init__(
        self,
        train_path   : str  = TRAIN_PROCESSED_PATH,
        test_path    : str  = TEST_PROCESSED_PATH,
        option_cols  : list = None,
        option_labels: list = None,
    ):
        self.train_path    = train_path
        self.test_path     = test_path
        self.option_cols   = option_cols   or OPTION_COLS
        self.option_labels = option_labels or OPTION_COLS  # ["A","B","C","D","E"]

    # ------------------------------------------------------------------
    # TEXT CLEANING
    # ------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', ' ', text)
        return text.strip()

    # ------------------------------------------------------------------
    # OPTION MAP BUILDER
    # ------------------------------------------------------------------
    @staticmethod
    def build_option_map(
        row          : pd.Series,
        option_cols  : List[str],
        option_labels: List[str],
    ) -> dict:
        return {
            label: TransformerDataLoader.clean_text(str(row[col]))
            for label, col in zip(option_labels, option_cols)
            if col in row.index
        }

    # ------------------------------------------------------------------
    # LOAD TRAIN
    # ------------------------------------------------------------------
    def load_train(self) -> pd.DataFrame:
        df = load_csv(self.train_path)
        print(f"Train shape : {df.shape}")

        # Clean prompt
        df["prompt_clean"] = df["prompt"].apply(self.clean_text)

        # Clean option columns in-place
        for col in self.option_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)

        # Standardize answer
        if ANSWER_COL in df.columns:
            df[ANSWER_COL] = df[ANSWER_COL].str.strip().str.upper()

        print(f"   Columns     : {list(df.columns)}")
        if ANSWER_COL in df.columns:
            print(f"   Answer dist :\n{df[ANSWER_COL].value_counts()}")

        # Save processed train file to working directory
        df.to_csv(TRAIN_OUTPUT_PATH, index=False)
        print(f"Processed train saved to: {TRAIN_OUTPUT_PATH}")

        return df

    # ------------------------------------------------------------------
    # LOAD TEST
    # ------------------------------------------------------------------
    def load_test(self) -> pd.DataFrame:
        df = load_csv(self.test_path)
        print(f"Test shape  : {df.shape}")

        df["prompt_clean"] = df["prompt"].apply(self.clean_text)
        for col in self.option_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        # Save processed test file to working directory
        df.to_csv(TEST_OUTPUT_PATH, index=False)
        print(f"Processed test saved to: {TEST_OUTPUT_PATH}")
        return df

    # ------------------------------------------------------------------
    # FORMAT FOR MODEL
    # -----------------------------------------------------------------
    def format_rows(self, df: pd.DataFrame) -> List[dict]:
        records = []
        for _, row in df.iterrows():
            option_map = self.build_option_map(
                row, self.option_cols, self.option_labels
            )
            # Filter out empty option texts
            option_map = {k: v for k, v in option_map.items() if v}

            records.append({
                "id"     : row.get(ID_COL, row.name),
                "prompt" : row["prompt_clean"],
                "options": option_map,
                "answer" : row.get(ANSWER_COL, None),
            })
        return records