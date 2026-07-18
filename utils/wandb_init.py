# wandb_setup/wandb_init.py
# Isolated W&B initialisation — keeps main.py clean.

import logging
from typing import Optional

from config.config import Config

logger = logging.getLogger("WandB")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def init_wandb(
    config  : Config,
    run_name: str = "phase1-baseline",
) -> Optional[object]:
    """
    Initialise a W&B run.

    Authentication order:
    1. Kaggle Secrets (``WANDB_API_KEY``)
    2. Environment variable / interactive login

    Returns the active run object, or None if W&B is disabled /
    authentication fails.

    Usage
    -----
        run = init_wandb(cfg, run_name="phase1-tfidf")
    """
    if not _WANDB_AVAILABLE:
        logger.warning("wandb package not installed — skipping tracking.")
        return None

    if not config.use_wandb:
        logger.info("W&B disabled in config.")
        return None

    try:
        # ── Kaggle Secrets ────────────────────────────────────
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
            wandb.login(key=key, relogin=True)
            logger.info("W&B authenticated via Kaggle Secrets.")
        except Exception:
            logger.info("Kaggle Secrets unavailable — trying env var / CLI.")
            wandb.login()

        run = wandb.init(
            project = config.wandb_project,
            entity  = config.wandb_entity,
            name    = run_name,
            config  = {
                "sbert_model"        : config.sbert_model,
                "tfidf_max_features" : config.tfidf_max_features,
                "tfidf_ngram_range"  : config.tfidf_ngram_range,
                "w2v_vector_size"    : config.w2v_vector_size,
                "top_k"              : config.top_k,
                "seed"               : config.seed,
                "device"             : config.device,
                "n_gpus"             : config.n_gpus,
            },
            reinit  = True,
            tags    = ["phase1", "baseline", "tfidf", "sbert"],
        )
        logger.info(f"W&B run: {run.url}")
        return run

    except Exception as exc:
        logger.warning(f"W&B init failed ({exc}) — continuing without tracking.")
        return None