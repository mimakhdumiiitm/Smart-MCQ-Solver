# utils/submission.py
# Kaggle submission file generation and validation.

import logging
from pathlib import Path
from typing import List

import pandas as pd

from config.config import Config

logger = logging.getLogger("Submission")


class SubmissionGenerator:
    """
    Build and validate a Kaggle-format submission CSV.

    Expected format
    ---------------
    id,prediction
    1,B A C
    2,A C D
    ...
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def generate(
        self,
        test_df    : pd.DataFrame,
        predictions: List[List[str]],
        filename   : str = "submission.csv",
    ) -> pd.DataFrame:
        """
        Create the submission DataFrame and write to disk.

        Parameters
        ----------
        test_df     : raw or processed test DataFrame (must have id column).
        predictions : top-K label lists aligned with test_df rows.
        filename    : output CSV name inside cfg.submission_dir.

        Returns
        -------
        pd.DataFrame with columns [id, prediction].
        """
        if len(test_df) != len(predictions):
            raise ValueError(
                f"Row count mismatch: "
                f"test_df={len(test_df)}, predictions={len(predictions)}"
            )

        sub = pd.DataFrame({
            self.cfg.id_col : test_df[self.cfg.id_col].values,
            "prediction"    : [" ".join(p) for p in predictions],
        })

        # Validate every row has exactly top_k labels
        pred_lengths = sub["prediction"].str.split().str.len()
        if pred_lengths.min() != self.cfg.top_k:
            bad = sub[pred_lengths != self.cfg.top_k]
            raise ValueError(
                f"Some predictions have wrong length:\n{bad.head()}"
            )

        out_path = Path(self.cfg.submission_dir) / filename
        sub.to_csv(out_path, index=False)
        logger.info(f"Submission saved → {out_path}  ({len(sub)} rows)")
        logger.info(f"Preview:\n{sub.head(3).to_string(index=False)}")
        return sub