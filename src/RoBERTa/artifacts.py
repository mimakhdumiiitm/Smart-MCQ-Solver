# src/RoBERTa/artifacts.py
"""
W&B artifact helpers for RoBERTa pipeline.

Changes from reviewed version
──────────────────────────────
  - Logging uses %-style formatting throughout.
  - _upload() is a no-op (with a debug log) when wandb is unavailable
    or run is None — unchanged semantics, cleaner log output.
  - save_model() / load_model() operate on the plain (unwrapped) model
    state dict, consistent with the Trainer's best_state convention.
  - No functional changes to artifact schema or file layout.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("RoBERTa.Artifacts")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_DEDUP = "mcq-roberta-dedup-data"
_MODEL = "mcq-roberta-model"
_AUDIT = "mcq-roberta-audit-report"
_SUB   = "mcq-roberta-submission"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _upload(
    run,
    name  : str,
    atype : str,
    desc  : str,
    files : Dict[Path, str],
    meta  : Optional[Dict] = None,
) -> None:
    if run is None or not _WANDB_AVAILABLE:
        logger.debug("W&B unavailable — skipping artifact upload '%s'.", name)
        return
    art = wandb.Artifact(
        name        = name,
        type        = atype,
        description = desc,
        metadata    = meta or {},
    )
    for local_path, artifact_name in files.items():
        art.add_file(str(local_path), name=artifact_name)
    run.log_artifact(art)
    logger.info("Artifact '%s' uploaded.", name)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def try_load(run, artifact_name: str, artifacts_load_dir: str):
    """
    Download *artifact_name* from W&B and return the Artifact object.
    Returns None silently if W&B is unavailable or the artifact is not found.
    """
    if run is None or not _WANDB_AVAILABLE:
        return None
    try:
        art = run.use_artifact(artifact_name)
        art.download(root=artifacts_load_dir)
        logger.info("Artifact '%s' restored.", artifact_name)
        return art
    except Exception as exc:
        logger.info("Artifact '%s' not found (%s).", artifact_name, exc)
        return None


def save_dedup(
    run,
    df    : pd.DataFrame,
    sbert : np.ndarray,
    artifacts_save_dir: str,
) -> None:
    d   = _dir(artifacts_save_dir)
    csv = d / "dedup_train.csv"
    npy = d / "sbert_matrix.npy"
    df.to_csv(csv, index=False)
    np.save(npy, sbert)
    _upload(
        run, _DEDUP, "dataset",
        "Deduplicated data + SBERT embeddings",
        {csv: "dedup_train.csv", npy: "sbert_matrix.npy"},
        {"n_rows": len(df), "sbert_shape": list(sbert.shape)},
    )


def load_dedup(artifacts_load_dir: str):
    d     = Path(artifacts_load_dir)
    df    = pd.read_csv(d / "dedup_train.csv")
    sbert = np.load(d / "sbert_matrix.npy")
    logger.info("Loaded dedup: %d rows, SBERT %s", len(df), sbert.shape)
    return df, sbert


def save_model(
    run,
    model,
    tokenizer,
    artifacts_save_dir : str,
    meta               : Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.

    Expects *model* to be the unwrapped MCQRoBERTa (not DataParallel).
    The state dict therefore has no "module." prefix and can be loaded
    directly with load_model() without key surgery.
    """
    import torch

    d    = _dir(artifacts_save_dir)
    ckpt = d / "roberta_best.pt"
    torch.save(model.state_dict(), ckpt)

    hf_dir = d / "hf_model"
    hf_dir.mkdir(exist_ok=True)
    model.encoder.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    files: Dict[Path, str] = {ckpt: "roberta_best.pt"}
    for f in hf_dir.rglob("*"):
        if f.is_file():
            files[f] = str(f.relative_to(d))

    _upload(
        run, _MODEL, "model",
        "RoBERTa-base best checkpoint",
        files,
        meta or {},
    )
    logger.info("Model saved to %s", d)


def load_model(model, artifacts_load_dir: str, device: str = "cpu"):
    """
    Load checkpoint into *model* (must be unwrapped MCQRoBERTa).
    Returns model on *device*.
    """
    import torch

    p = Path(artifacts_load_dir) / "roberta_best.pt"
    model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device)


def save_audit(run, report: dict, artifacts_save_dir: str) -> None:
    d = _dir(artifacts_save_dir)
    p = d / "audit_report.json"

    def _serialise(obj):
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(p, "w") as fh:
        json.dump(_serialise(report), fh, indent=2)

    _upload(
        run, _AUDIT, "report",
        "Leakage audit report",
        {p: "audit_report.json"},
    )


def save_submission(run, sub: pd.DataFrame, submission_dir: str) -> None:
    d = _dir(submission_dir)
    p = d / "RoBERTa_submission.csv"
    sub.to_csv(p, index=False)
    _upload(
        run, _SUB, "predictions",
        "RoBERTa final submission",
        {p: "RoBERTa_submission.csv"},
        {"n_rows": len(sub)},
    )