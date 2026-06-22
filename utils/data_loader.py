# data_loader.py

import os
import pandas as pd
import numpy as np

from config.config import (
    TRAIN_PATH, TEST_PATH, SUBMISSION_PATH,
    ID_COL, PROMPT_COL, ANSWER_COL, OPTION_COLS, TEXT_COLS
)


# ------------------------------------------------------------------
# CORE LOADER
# ------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    Raises FileNotFoundError with a helpful message if not found.
    
    Usage:
        df = load_csv("data/train.csv")
    """
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


# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame,
                        required_cols: list,
                        name: str = "DataFrame") -> bool:
    """
    Check that all required columns exist in a DataFrame.
    
    Returns True if valid, raises ValueError otherwise.
    
    Usage:
        validate_dataframe(train_df, TEXT_COLS + [ANSWER_COL], "Train")
    """
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{name} is missing columns: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    print(f"{name} validation passed. All required columns present.")
    return True


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of missing values per column.
    
    Usage:
        missing_report = check_missing_values(train_df)
        print(missing_report)
    """
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
    """
    Fill missing values in text columns with a placeholder string.
    
    Usage:
        df = fill_missing_text(train_df, TEXT_COLS, fill_value="")
    """
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


# ------------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------------

def basic_stats(df: pd.DataFrame) -> dict:
    """
    Compute and return a dict of basic dataset statistics.
    
    Usage:
        stats = basic_stats(train_df)
        for k, v in stats.items():
            print(k, v)
    """
    stats = {
        "n_rows"         : len(df),
        "n_cols"         : len(df.columns),
        "columns"        : df.columns.tolist(),
        "dtypes"         : df.dtypes.to_dict(),
        "memory_kb"      : df.memory_usage(deep=True).sum() / 1024,
        "total_missing"  : df.isnull().sum().sum(),
    }

    if ANSWER_COL in df.columns:
        stats["answer_distribution"] = df[ANSWER_COL].value_counts().to_dict()

    return stats


def print_basic_stats(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Pretty-print basic dataset statistics.
    
    Usage:
        print_basic_stats(train_df, "Train")
    """
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