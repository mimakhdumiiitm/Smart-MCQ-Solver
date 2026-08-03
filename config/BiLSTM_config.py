# config.py
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class Config:
    # paths
    train_path   : str            = 'train.csv'
    test_path    : Optional[str]  = None
    artifact_dir : str            = './artifacts'

    # W&B
    use_wandb      : bool         = True
    wandb_project  : str          = 'mcq-bilstm'
    wandb_entity   : Optional[str]= None
    wandb_run_name : str          = 'bilstm-run'

    # dedup / split
    sbert_model   : str   = 'all-MiniLM-L6-v2'
    sim_threshold : float = 0.85
    val_size      : float = 0.15
    seed          : int   = 42

    # vocab
    max_vocab : int = 20_000
    min_freq  : int = 2

    # model
    embed_dim : int   = 100
    hidden    : int   = 128
    n_layers  : int   = 2
    dropout   : float = 0.4
    max_len   : int   = 180

    # training
    batch_size    : int   = 64
    epochs        : int   = 40
    lr            : float = 5e-4
    weight_decay  : float = 1e-3
    max_grad_norm : float = 0.5

    # loss
    smoothing : float = 0.15
    margin    : float = 0.5
    ce_w      : float = 0.6
    rank_w    : float = 0.4

    # scheduler
    sched_patience : int   = 4
    sched_factor   : float = 0.5

    # early stop
    early_stop_patience : int = 8

    # audit
    audit_top_k : int = 20

    # runtime — auto-set
    device : str = 'cpu'
    n_gpus : int = 0

    # required by wandb_init.py
    top_k              : int   = 3
    tfidf_max_features : int   = 30_000
    tfidf_ngram_range  : Tuple = (1, 2)
    w2v_vector_size    : int   = 100

    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gpus = torch.cuda.device_count()