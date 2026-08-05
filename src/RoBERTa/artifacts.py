# src/RoBERTa/artifacts.py
"""
W&B artifact helpers for RoBERTa pipeline.
Mirrors DeBERTa artifacts.py with RoBERTa-specific names.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("RoBERTa.Artifacts")

try:
    import wandb
    _W = True
except ImportError:
    _W = False

_DEDUP = "mcq-roberta-dedup-data"
_MODEL = "mcq-roberta-model"
_AUDIT = "mcq-roberta-audit-report"
_SUB   = "mcq-roberta-submission"


def _dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _upload(run, name, atype, desc, files: dict, meta: dict = None):
    if run is None or not _W:
        return
    art = wandb.Artifact(name=name, type=atype,
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


def save_dedup(run, df: pd.DataFrame,
               sbert: np.ndarray, artifacts_save_dir: str):
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
    logger.info(f"Loaded dedup: {len(df)} rows, SBERT {sbert.shape}")
    return df, sbert


def save_model(run, model, tokenizer,
               artifacts_save_dir: str, meta: dict = None):
    import torch
    d    = _dir(artifacts_save_dir)
    ckpt = d / "roberta_best.pt"
    torch.save(model.state_dict(), ckpt)

    hf_dir = d / "hf_model"
    hf_dir.mkdir(exist_ok=True)
    model.encoder.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    files = {ckpt: "roberta_best.pt"}
    for f in hf_dir.rglob("*"):
        if f.is_file():
            files[f] = str(f.relative_to(d))

    _upload(run, _MODEL, "model",
            "RoBERTa-base best checkpoint", files, meta or {})
    logger.info(f"Model saved to {d}")


def load_model(model, artifacts_load_dir: str, device: str = "cpu"):
    import torch
    p = Path(artifacts_load_dir) / "roberta_best.pt"
    model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device)


def save_audit(run, report: dict, artifacts_save_dir: str):
    d = _dir(artifacts_save_dir)
    p = d / "audit_report.json"

    def _s(o):
        if isinstance(o, dict):                       return {k: _s(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):              return [_s(v) for v in o]
        if isinstance(o, set):                        return list(o)
        if isinstance(o, (np.integer, np.floating)):  return o.item()
        if isinstance(o, np.ndarray):                 return o.tolist()
        return o

    with open(p, "w") as f:
        json.dump(_s(report), f, indent=2)
    _upload(run, _AUDIT, "report",
            "Leakage audit report", {p: "audit_report.json"})


def save_submission(run, sub: pd.DataFrame, submission_dir: str):
    d = _dir(submission_dir)
    p = d / "RoBERTa_submission.csv"
    sub.to_csv(p, index=False)
    _upload(run, _SUB, "predictions", "RoBERTa final submission",
            {p: "RoBERTa_submission.csv"}, {"n_rows": len(sub)})