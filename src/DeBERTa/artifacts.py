# src/DeBERTa/artifacts.py
"""
W&B artifact helpers for DeBERTa.

Differences vs. BiLSTM artifacts.py
────────────────────────────────────
• Model checkpoint saved as  deberta_best.pt
• No Vocabulary artifact  (tokenizer comes from HuggingFace hub)
• Tokenizer saved locally with save_pretrained  (for offline inference)
• Everything else (dedup, audit, submission) reused verbatim
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("DeBERTa.Artifacts")

try:
    import wandb
    _W = True
except ImportError:
    _W = False

# ── artifact names (separate namespace from BiLSTM) ──────────────────────────
_DEDUP = "mcq-dedup-data"          # shared with BiLSTM (same data)
_MODEL = "mcq-deberta-model"
_AUDIT = "mcq-deberta-audit"
_SUB   = "mcq-deberta-submission"
_TOK   = "mcq-deberta-tokenizer"


def _dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _upload(run, name, atype, desc, files: dict, meta: dict = None):
    if run is None or not _W:
        return
    art = wandb.Artifact(
        name=name, type=atype,
        description=desc, metadata=meta or {})
    for local, aname in files.items():
        art.add_file(str(local), name=aname)
    run.log_artifact(art)
    logger.info(f"Artifact '{name}' uploaded.")


def try_load(run, artifact_name: str, artifacts_load_dir: str):
    if run is None or not _W:
        return None
    try:
        art = run.use_artifact(artifact_name)
        art.download(root=artifacts_load_dir)
        logger.info(f"Artifact '{artifact_name}' restored.")
        return art
    except Exception as e:
        logger.info(f"Artifact '{artifact_name}' not found ({e}).")
        return None


# ── reuse dedup helpers from BiLSTM ──────────────────────────────────────────
# (same data, same BoW pipeline — no need to duplicate)

from src.BiLSTM.artifacts import (
    save_dedup, load_dedup,
    save_audit,
)


# ── tokenizer ─────────────────────────────────────────────────────────────────

def save_tokenizer(run, tokenizer, artifacts_save_dir: str, model_name: str):
    """
    Save tokenizer files locally AND upload to W&B.
    Allows fully offline inference without HuggingFace hub.
    """
    d = _dir(Path(artifacts_save_dir) / "tokenizer")
    tokenizer.save_pretrained(str(d))

    files = {f: f.name for f in d.iterdir() if f.is_file()}
    _upload(run, _TOK, "tokenizer",
            f"DeBERTa tokenizer ({model_name})",
            files, {"model_name": model_name})


def load_tokenizer_local(artifacts_load_dir: str):
    from transformers import AutoTokenizer
    d = Path(artifacts_load_dir) / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(d))
    logger.info(f"Tokenizer loaded from local cache: {d}")
    return tok


# ── model ─────────────────────────────────────────────────────────────────────

def save_model(run, model, artifacts_save_dir: str, meta: dict = None):
    import torch
    d = _dir(artifacts_save_dir)
    p = d / "deberta_best.pt"
    torch.save(model.state_dict(), p)
    _upload(run, _MODEL, "model",
            "Best DeBERTa-v3 checkpoint",
            {p: "deberta_best.pt"}, meta or {})


def load_model(model, artifacts_load_dir: str, device: str = "cpu"):
    import torch
    p = Path(artifacts_load_dir) / "deberta_best.pt"
    model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device)


# ── submission ────────────────────────────────────────────────────────────────

def save_submission(run, sub: pd.DataFrame, submission_dir: str):
    d = _dir(submission_dir)
    p = d / "DeBERTa_submission.csv"
    sub.to_csv(p, index=False)
    _upload(run, _SUB, "predictions",
            "DeBERTa final submission",
            {p: "DeBERTa_submission.csv"},
            {"n_rows": len(sub)})