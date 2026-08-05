# utils/wandb_init.py
"""
W&B helpers — model-agnostic version.
Accepts any config object that has the required attributes.
"""

import logging
from typing import Optional, Dict, Any, List

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


def authenticate() -> None:
    """
    Authenticate with W&B.
    Priority:
      1. Kaggle Secrets  (WANDB_API_KEY)
      2. WANDB_API_KEY   environment variable
      3. ~/.netrc
    Never falls back to interactive prompt.
    """
    if not _WANDB_AVAILABLE:
        return

    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        if key:
            wandb.login(key=key, relogin=True)
            logger.info("W&B authenticated via Kaggle Secrets.")
            return
    except Exception:
        pass

    try:
        result = wandb.login(anonymous="never", relogin=False)
        if result:
            logger.info("W&B authenticated via env var / netrc.")
        else:
            logger.warning(
                "W&B login returned False. "
                "Set WANDB_API_KEY in environment or Kaggle Secrets."
            )
    except Exception as exc:
        logger.warning(f"W&B login failed: {exc}")


_authenticate = authenticate   # backward-compat alias


def init_wandb(
    config,                        # any config dataclass
    run_name   : str,
    model_name : str,
    group      : str               = "model-comparison",
    tags       : Optional[List[str]] = None,
) -> Optional[object]:
    """Initialise a single W&B run for one model."""
    if not _WANDB_AVAILABLE:
        logger.warning("wandb not installed — skipping.")
        return None

    if not getattr(config, 'use_wandb', False):
        logger.info("W&B disabled in config.")
        return None

    try:
        authenticate()

        run_config: Dict[str, Any] = {
            "model"  : model_name,
            "top_k"  : getattr(config, 'top_k',  3),
            "seed"   : getattr(config, 'seed',    42),
            "device" : getattr(config, 'device',  "cpu"),
            "n_gpus" : getattr(config, 'n_gpus',  1),
            **{m: None for m in REQUIRED_METRICS},
        }

        run = wandb.init(
            project  = config.wandb_project,
            entity   = getattr(config, 'wandb_entity', None) or None,
            name     = run_name,
            config   = run_config,
            group    = group,
            job_type = model_name,
            reinit   = True,
            tags     = tags or [model_name],
        )

        logger.info(f"W&B run | model={model_name} | url={run.url}")
        return run

    except Exception as exc:
        logger.warning(f"W&B init failed ({exc}) — skipping.")
        return None


def log_model_metrics(
    run    : Optional[object],
    metrics: Dict[str, float],
    step   : Optional[int] = None,
) -> None:
    if run is None or not _WANDB_AVAILABLE:
        return

    missing = [m for m in REQUIRED_METRICS if m not in metrics]
    if missing:
        logger.warning(f"[{run.name}] Missing required metrics: {missing}")

    log_kwargs: Dict[str, Any] = dict(metrics)
    if step is not None:
        log_kwargs["step"] = step

    run.log(log_kwargs)
    logger.debug(f"[{run.name}] Logged: {list(metrics.keys())}")


def finish_run(run: Optional[object]) -> None:
    if run is not None and _WANDB_AVAILABLE:
        run.finish()
        logger.info(f"W&B run '{run.name}' finished.")