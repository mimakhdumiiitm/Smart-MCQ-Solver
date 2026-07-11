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