class SubmissionGenerator:
    def __init__(self, config: Config):
        self.cfg = config

    def generate(
        self,
        test_df: pd.DataFrame,
        predictions: List[List[str]],
        filename: str = "submission.csv",
    ) -> pd.DataFrame:

        if len(test_df) != len(predictions):
            raise ValueError(
                f"Row count mismatch: "
                f"test_df={len(test_df)}, predictions={len(predictions)}"
            )

        sub = pd.DataFrame({
            self.cfg.id_col: test_df[self.cfg.id_col].values,
            "prediction": [" ".join(p) for p in predictions],
        })

        pred_lengths = sub["prediction"].str.split().str.len()
        if pred_lengths.min() != self.cfg.top_k:
            bad = sub[pred_lengths != self.cfg.top_k]
            raise ValueError(
                f"Some predictions have wrong length:\n{bad.head()}"
            )

        out_path = Path(self.cfg.submission_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sub.to_csv(out_path, index=False)

        logger.info(f"Submission saved → {out_path} ({len(sub)} rows)")
        logger.info(f"Preview:\n{sub.head(3).to_string(index=False)}")

        return sub