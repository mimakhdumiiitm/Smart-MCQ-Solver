<<<<<<< HEAD
# cell 6
# utils/submission.py

import pandas as pd
import wandb

from config.config import (
    OPTION_COLS,
    SUBMISSION_OUT_PATH,
    TOP_K,
)


class SubmissionGenerator:
    def __init__(
        self,
        option_labels   : list = None,
        top_k           : int  = TOP_K,
        output_path     : str  = SUBMISSION_OUT_PATH,
    ):
        self.option_labels = option_labels or OPTION_COLS
        self.top_k         = top_k
        self.output_path   = output_path

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    def generate(self, test_results: list) -> pd.DataFrame:
        rows = []
        for res in test_results:
            preds = list(res["top_labels"])

            # Pad to top_k if needed (e.g., fewer than 5 options in row)
            for label in self.option_labels:
                if len(preds) >= self.top_k:
                    break
                if label not in preds:
                    preds.append(label)

            rows.append({
                "ID"        : res["id"],
                "Prediction": " ".join(preds[: self.top_k]),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------------

    def validate(self, sub_df: pd.DataFrame) -> bool:
        print("\nValidating submission...")
        valid = True

        # Column check
        assert "ID"         in sub_df.columns, "Missing 'ID' column"
        assert "Prediction" in sub_df.columns, "Missing 'Prediction' column"

        for _, row in sub_df.iterrows():
            preds = row["Prediction"].split()

            # Prediction count check
            if len(preds) != self.top_k:
                print(f"  ID {row['ID']}: has {len(preds)} predictions "
                      f"(expected {self.top_k})")
                valid = False

            # Valid label check
            for p in preds:
                if p not in self.option_labels:
                    print(f"  ID {row['ID']}: invalid label '{p}'")
                    valid = False

        if valid:
            print(f"Submission valid! {len(sub_df)} rows, "
                  f"{self.top_k} predictions each.")
        return valid

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(self, sub_df: pd.DataFrame) -> str:
        sub_df.to_csv(self.output_path, index=False)
        print(f"Submission saved: {self.output_path}")

        # W&B logging
        wandb.log({
            "submission_rows": len(sub_df),
            "submission_file": wandb.Table(dataframe=sub_df.head(10)),
        })
        wandb.save(self.output_path)

        return self.output_path
=======
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
>>>>>>> Milestone-1
