# utils/wandb_init.py

import logging
from typing import Optional, Dict, Any, List

from config.config import Config

logger = logging.getLogger("WandB")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

REQUIRED_METRICS: List[str] = [
    "f1_score",
    "accuracy",
    "precision",
    "recall",
    "map_at_k",
]

COMPARABLE_MODELS: List[str] = ["tfidf", "word2vec", "sbert"]


def _authenticate() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=key, relogin=True)
        logger.info("W&B authenticated via Kaggle Secrets.")
    except Exception:
        logger.info("Kaggle Secrets unavailable — trying env var / CLI.")
        wandb.login()


def init_wandb(
    config   : Config,
    run_name : str = "phase1-baseline",
    model_tag: str = "tfidf",
) -> Optional[object]:
    """
    Initialise a single W&B run for one model.
    model_tag must be one of: tfidf, word2vec, sbert
    """
    if not _WANDB_AVAILABLE:
        logger.warning("wandb not installed — skipping.")
        return None

    if not config.use_wandb:
        logger.info("W&B disabled in config.")
        return None

    try:
        _authenticate()

        run_config: Dict[str, Any] = {
            "model"              : model_tag,
            "top_k"              : config.top_k,
            "seed"               : config.seed,
            "device"             : config.device,
            "n_gpus"             : config.n_gpus,
            "sbert_model"        : config.sbert_model,
            "tfidf_max_features" : config.tfidf_max_features,
            "tfidf_ngram_range"  : config.tfidf_ngram_range,
            "w2v_vector_size"    : config.w2v_vector_size,
            **{m: None for m in REQUIRED_METRICS},
        }

        run = wandb.init(
            project  = config.wandb_project,
            entity   = config.wandb_entity,
            name     = run_name,
            config   = run_config,
            group    = "phase1-model-comparison",   # ties all 3 runs together
            job_type = model_tag,
            reinit   = True,
            tags     = ["phase1", "baseline", model_tag],
        )

        logger.info(f"W&B run | model={model_tag} | url={run.url}")
        return run

    except Exception as exc:
        logger.warning(f"W&B init failed ({exc}) — skipping.")
        return None


def log_model_metrics(
    run    : Optional[object],
    metrics: Dict[str, float],
    step   : Optional[int] = None,
) -> None:
    """
    Log metrics to a run. Warns if required comparison metrics are missing.

    Usage
    -----
        log_model_metrics(run, {
            "f1_score" : 0.87,
            "accuracy" : 0.89,
            "precision": 0.86,
            "recall"   : 0.88,
            "map_at_k" : 0.91,
        })
    """
    if run is None or not _WANDB_AVAILABLE:
        return

    missing = [m for m in REQUIRED_METRICS if m not in metrics]
    if missing:
        logger.warning(
            f"[{run.name}] Missing required metrics: {missing}"
        )

    log_kwargs: Dict[str, Any] = dict(metrics)
    if step is not None:
        log_kwargs["step"] = step

    run.log(log_kwargs)
    logger.debug(f"[{run.name}] Logged: {list(metrics.keys())}")


def finish_run(run: Optional[object]) -> None:
    """Finish a single W&B run."""
    if run is not None and _WANDB_AVAILABLE:
        run.finish()
        logger.info(f"W&B run '{run.name}' finished.")