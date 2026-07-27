# src/utils/submission.py
import logging
from pathlib import Path
from typing import List

import pandas as pd

from config.config import Config

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    Convert top-K predictions into a Kaggle submission CSV.

    Expected output format:
        id,prediction
        0001,A B C
        0002,C A B
        ...

    Each prediction is a space-joined string of top_k option letters
    ordered by model confidence (most confident first).
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self,
        test_df: pd.DataFrame,
        predictions: List[List[str]],
    ) -> pd.DataFrame:
        """
        Args:
            test_df    : Test DataFrame (must contain id_col)
            predictions: List of top-K option lists
                         e.g. [["A","C","B"], ...]

        Returns:
            pd.DataFrame with columns [id, prediction]
        """

        if len(test_df) != len(predictions):
            raise ValueError(
                f"Row count mismatch: "
                f"test_df={len(test_df)}, predictions={len(predictions)}"
            )

        ids = test_df[self.config.id_col].astype(str).tolist()

        # Ensure each prediction has exactly top_k elements
        processed_preds = []
        for p in predictions:
            if len(p) < self.config.top_k:
                raise ValueError(
                    f"Prediction length {len(p)} < top_k={self.config.top_k}: {p}"
                )
            processed_preds.append(" ".join(p[: self.config.top_k]))

        submission = pd.DataFrame(
            {
                self.config.id_col: ids,
                "prediction": processed_preds,
            }
        )

        logger.info(
            f"Submission built: {len(submission)} rows | "
            f"Preview:\n{submission.head(3).to_string(index=False)}"
        )

        return submission

    def save(
        self,
        submission: pd.DataFrame,
        filename: str = "submission.csv",
    ) -> Path:
        """Save submission to submission_dir/{filename}."""

        output_dir = Path(self.config.submission_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / filename
        submission.to_csv(path, index=False)

        logger.info(f"Submission saved → {path}")
        return path