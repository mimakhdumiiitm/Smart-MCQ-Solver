# cell 2
# utils/wandb_utils.py

import os
import wandb

from config.config import (
    WANDB_PROJECT,
    WANDB_RUN,
    WANDB_TAGS,
    WANDB_SECRET_KEY_NAME,
    EMBEDDING_MODEL,
    ZEROSHOT_MODEL,
    STRATEGY,
    ENSEMBLE_WEIGHTS,
    TRANSFORMER_BATCH_SIZE,
    MAX_SEQ_LENGTH,
    TOP_K,
)


# ------------------------------------------------------------------
# AUTHENTICATION
# ------------------------------------------------------------------

def authenticate_wandb(secret_key_name: str = WANDB_SECRET_KEY_NAME) -> bool:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_key    = user_secrets.get_secret(secret_key_name)
        wandb.login(key=wandb_key)
        print("W&B authenticated via Kaggle secrets")
        return True
    except Exception as e:
        print(f"W&B auth warning: {e} — running in offline mode")
        os.environ["WANDB_MODE"] = "offline"
        return False


# ------------------------------------------------------------------
# RUN INITIALIZATION
# ------------------------------------------------------------------

def init_wandb_run(
    project          : str  = WANDB_PROJECT,
    run_name         : str  = WANDB_RUN,
    embedding_model  : str  = EMBEDDING_MODEL,
    zeroshot_model   : str  = ZEROSHOT_MODEL,
    strategy         : str  = STRATEGY,
    ensemble_weights : dict = None,
    batch_size       : int  = TRANSFORMER_BATCH_SIZE,
    max_seq_length   : int  = MAX_SEQ_LENGTH,
    top_k            : int  = TOP_K,
    tags             : list = None,
) -> wandb.run:
    if ensemble_weights is None:
        ensemble_weights = ENSEMBLE_WEIGHTS
    if tags is None:
        tags = WANDB_TAGS

    run = wandb.init(
        project = project,
        name    = run_name,
        config  = {
            "stage"            : "Transformer Embeddings + Zero-Shot",
            "embedding_model"  : embedding_model,
            "zeroshot_model"   : zeroshot_model,
            "strategy"         : strategy,
            "ensemble_weights" : ensemble_weights,
            "batch_size"       : batch_size,
            "max_seq_length"   : max_seq_length,
            "top_k"            : top_k,
        },
        tags = tags,
    )

    print(f"W&B run initialized : {run.name}")
    print(f"Project             : {project}")
    return run


# ------------------------------------------------------------------
# FULL SETUP ENTRY POINT
# ------------------------------------------------------------------
def setup_wandb(secret_key_name: str = WANDB_SECRET_KEY_NAME) -> wandb.run:
    authenticate_wandb(secret_key_name)
    run = init_wandb_run()
    print("W&B setup complete")
    return run


# ------------------------------------------------------------------
# SAFE FINISH
# ------------------------------------------------------------------

def finish_wandb_run(
    run,
    final_metrics : dict = None,
    device_used   : str  = "",
) -> None:
    summary_update = {"device_used": device_used}
    if final_metrics:
        summary_update.update(final_metrics)

    run.summary.update(summary_update)
    wandb.finish()
    print("W&B run finalized")