# artifacts.py
"""
W&B artifact save / restore helpers.
Five artifacts: dedup-data, vocab, model, audit-report, submission.
BoW matrix stored as .npy (replaces SBERT embeddings).
"""

import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("Artifacts")

try:
    import wandb
    _W = True
except ImportError:
    _W = False

_DEDUP = "mcq-dedup-data"
_VOCAB = "mcq-vocab"
_MODEL = "mcq-model"
_AUDIT = "mcq-audit-report"
_SUB   = "mcq-submission"


def _dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def _upload(run, name, atype, desc, files: dict, meta: dict = None):
    if run is None or not _W:
        return
    art = wandb.Artifact(name=name, type=atype,
                         description=desc, metadata=meta or {})
    for local, aname in files.items():
        art.add_file(str(local), name=aname)
    run.log_artifact(art)
    logger.info(f"Artifact '{name}' uploaded.")


def try_load(run, artifact_name: str, artifact_dir: str):
    if run is None or not _W:
        return None
    try:
        art = run.use_artifact(artifact_name)
        art.download(root=artifact_dir)
        logger.info(f"Artifact '{artifact_name}' restored.")
        return art
    except Exception as e:
        logger.info(f"Artifact '{artifact_name}' not found ({e}).")
        return None


# ── dedup + BoW matrix ───────────────────────────────────────────────────────

def save_dedup(run, df: pd.DataFrame, bow: np.ndarray, artifact_dir: str):
    d   = _dir(artifact_dir)
    csv = d / "dedup_train.csv";  df.to_csv(csv, index=False)
    npy = d / "bow_matrix.npy";   np.save(npy, bow)
    _upload(run, _DEDUP, "dataset",
            "Deduplicated data + BoW matrix",
            {csv: "dedup_train.csv", npy: "bow_matrix.npy"},
            {"n_rows": len(df), "bow_shape": list(bow.shape)})


def load_dedup(artifact_dir: str):
    d   = Path(artifact_dir)
    df  = pd.read_csv(d / "dedup_train.csv")
    bow = np.load(d / "bow_matrix.npy")
    logger.info(f"Loaded dedup: {len(df)} rows, BoW {bow.shape}")
    return df, bow


# ── vocab ─────────────────────────────────────────────────────────────────────

def save_vocab(run, vocab, artifact_dir: str):
    d = _dir(artifact_dir)
    p = d / "vocabulary.pkl"
    with open(p, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    _upload(run, _VOCAB, "vocab", "Fitted LSTM Vocabulary",
            {p: "vocabulary.pkl"}, {"vocab_size": len(vocab)})


def load_vocab(artifact_dir: str):
    with open(Path(artifact_dir) / "vocabulary.pkl", "rb") as f:
        return pickle.load(f)


# ── model ─────────────────────────────────────────────────────────────────────

def save_model(run, model, artifact_dir: str, meta: dict = None):
    import torch
    d = _dir(artifact_dir)
    p = d / "bilstm_best.pt"
    torch.save(model.state_dict(), p)
    _upload(run, _MODEL, "model", "Best Bi-LSTM checkpoint",
            {p: "bilstm_best.pt"}, meta or {})


def load_model(model, artifact_dir: str, device: str = "cpu"):
    import torch
    p = Path(artifact_dir) / "bilstm_best.pt"
    model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device)


# ── audit ─────────────────────────────────────────────────────────────────────

def save_audit(run, report: dict, artifact_dir: str):
    d = _dir(artifact_dir)
    p = d / "audit_report.json"

    def _s(o):
        if isinstance(o, dict):           return {k: _s(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):  return [_s(v) for v in o]
        if isinstance(o, set):            return list(o)
        if isinstance(o, (np.integer, np.floating)): return o.item()
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    with open(p, "w") as f:
        json.dump(_s(report), f, indent=2)
    _upload(run, _AUDIT, "report", "Leakage audit report",
            {p: "audit_report.json"})


# ── submission ────────────────────────────────────────────────────────────────

def save_submission(run, sub: pd.DataFrame, artifact_dir: str):
    d = _dir(artifact_dir)
    p = d / "BiLSTM_submission.csv" 
    sub.to_csv(p, index=False)
    _upload(
        run,
        _SUB,
        "predictions",
        "Final submission",
        {p: "BiLSTM_submission.csv"},  
        {"n_rows": len(sub)}
    )