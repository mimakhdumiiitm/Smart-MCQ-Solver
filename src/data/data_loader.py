# data/data_loader.py
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from config.config import Config

logger = logging.getLogger("DataLoader")


class DataLoader:
    """
    Handles all raw data I/O.

    If a real CSV is missing (local dev / CI), falls back to synthetic
    data so every downstream cell can still run end-to-end.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def load_train(self) -> pd.DataFrame:
        """Load the raw training CSV (or generate synthetic fallback)."""
        return self._load(self.cfg.train_path, include_answer=True, n_synth=1000)

    def load_test(self) -> pd.DataFrame:
        """Load the raw test CSV (or generate synthetic fallback)."""
        return self._load(self.cfg.test_path, include_answer=False, n_synth=200)

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _load(
        self,
        path: Path,
        include_answer: bool,
        n_synth: int,
    ) -> pd.DataFrame:
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded {path.name}: {df.shape}")
            return df

        logger.warning(f"{path} not found — generating synthetic data.")
        return self._generate_synthetic(n_synth, include_answer)

    def _generate_synthetic(
        self, n_samples: int, include_answer: bool
    ) -> pd.DataFrame:
        """
        Produce synthetic MCQ rows for local development.
        Options are shuffled per row so the correct letter varies.
        """
        np.random.seed(self.cfg.seed)

        templates = [
            ("What is the speed of light?",
             ["299,792,458 m/s", "300,000 km/h", "150,000 m/s",
              "3×10^8 km/s", "186,000 mph"], "A"),
            ("Which planet is closest to the Sun?",
             ["Venus", "Mercury", "Earth", "Mars", "Jupiter"], "B"),
            ("What is the powerhouse of the cell?",
             ["Nucleus", "Ribosome", "Mitochondria",
              "Golgi apparatus", "Lysosome"], "C"),
            ("Who developed the theory of general relativity?",
             ["Newton", "Bohr", "Hawking", "Einstein", "Tesla"], "D"),
            ("What is H2O commonly known as?",
             ["Hydrogen peroxide", "Ammonia", "Methane",
              "Carbon dioxide", "Water"], "E"),
        ]

        rows = []
        for i in range(n_samples):
            prompt, opts, correct_letter = templates[i % len(templates)]
            opts = opts.copy()
            correct_text = opts[ord(correct_letter) - ord("A")]

            np.random.shuffle(opts)
            new_letter = chr(ord("A") + opts.index(correct_text))

            row: dict = {
                "id": i + 1, "prompt": prompt,
                "A": opts[0], "B": opts[1], "C": opts[2],
                "D": opts[3], "E": opts[4],
            }
            if include_answer:
                row["answer"] = new_letter
            rows.append(row)

        return pd.DataFrame(rows)