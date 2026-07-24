# utils/persistence.py
# Centralised save / load helpers for models AND processed DataFrames.
# Every ranker that can be serialised calls these helpers — zero duplication.

import pickle
import logging
from pathlib import Path
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger("Persistence")


# ─────────────────────────────────────────────
# Generic pickle helpers
# ─────────────────────────────────────────────

def save_pickle(obj: Any, path: Path) -> None:
    """
    Serialise *obj* to *path* using pickle.

    Usage:
        save_pickle(vectorizer, cfg.tfidf_model_path)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"[pickle] saved  → {path}  ({path.stat().st_size / 1e3:.1f} KB)")


def load_pickle(path: Path) -> Optional[Any]:
    """
    Deserialise a pickle file.

    Returns None and logs a warning if the file does not exist.

    Usage:
        vectorizer = load_pickle(cfg.tfidf_model_path)
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"[pickle] file not found: {path}")
        return None
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    logger.info(f"[pickle] loaded ← {path}")
    return obj


# ─────────────────────────────────────────────
# Gensim Word2Vec helpers
# ─────────────────────────────────────────────

def save_w2v(model: Any, path: Path) -> None:
    """
    Save a Gensim Word2Vec model in native binary format.

    Usage:
        save_w2v(w2v_model, cfg.w2v_model_path)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info(f"[Word2Vec] saved  → {path}")


def load_w2v(path: Path) -> Optional[Any]:
    """
    Load a Gensim Word2Vec model.

    Returns None if file is missing.

    Usage:
        model = load_w2v(cfg.w2v_model_path)
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"[Word2Vec] file not found: {path}")
        return None
    try:
        from gensim.models import Word2Vec
        model = Word2Vec.load(str(path))
        logger.info(f"[Word2Vec] loaded ← {path}")
        return model
    except Exception as exc:
        logger.error(f"[Word2Vec] failed to load: {exc}")
        return None


# ─────────────────────────────────────────────
# Processed DataFrame helpers
# ─────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save a processed DataFrame to CSV.

    Usage:
        save_dataframe(train_df, cfg.processed_train_path)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        f"[DataFrame] saved  → {path}  "
        f"({len(df):,} rows × {len(df.columns)} cols, "
        f"{path.stat().st_size / 1e6:.2f} MB)"
    )


def load_dataframe(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a processed DataFrame from CSV.

    Returns None if the file does not exist.

    Usage:
        train_df = load_dataframe(cfg.processed_train_path)
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"[DataFrame] file not found: {path}")
        return None
    df = pd.read_csv(path)
    logger.info(
        f"[DataFrame] loaded ← {path}  "
        f"({len(df):,} rows × {len(df.columns)} cols)"
    )
    return df