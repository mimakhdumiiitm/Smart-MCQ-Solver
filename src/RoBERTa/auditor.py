# src/RoBERTa/auditor.py
"""
Leakage auditor — SBERT cosine similarity post-split check.

Changes from reviewed version
──────────────────────────────
  - run() now returns max_sims (the per-val-sample maximum cosine
    similarity to any training sample) inside the report dict under
    the key 'max_sims'.  The pipeline reuses this value instead of
    recomputing the full similarity matrix a second time.
  - Similarity matrix computed once; max_sims derived from it in-place
    (no duplicate cosine_similarity call).
  - _top_pairs vectorised: avoids repeated iloc calls by using
    pre-fetched numpy arrays.
  - Logging uses %-style formatting (consistent with stdlib logger).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.RoBERTa.data import normalize_text

logger = logging.getLogger("RoBERTa.Auditor")


class LeakageAuditor:
    """Post-split leakage audit using pre-computed SBERT vectors."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        train_df    : pd.DataFrame,
        val_df      : pd.DataFrame,
        train_sbert : np.ndarray,
        val_sbert   : np.ndarray,
        wandb_run   = None,
    ) -> dict:
        """
        Run the audit and return a report dict.

        The returned dict includes 'max_sims' (np.ndarray of shape
        [n_val]) so the pipeline can reuse it for plotting without
        recomputing the full similarity matrix.
        """
        logger.info("─" * 60)
        logger.info("LEAKAGE AUDIT  (SBERT cosine)")
        logger.info("─" * 60)
        report: dict = {}

        # A — exact fingerprint overlap
        for col, label in [
            ("exact_fp",  "question+options"),
            ("option_fp", "option-set only"),
        ]:
            overlap = (
                set(train_df.get(col, pd.Series(dtype=str))) &
                set(val_df.get(col,   pd.Series(dtype=str)))
            )
            report[col] = len(overlap)
            logger.info("Exact %s overlap: %d", label, len(overlap))

        # B — SBERT cosine similarity (computed ONCE)
        logger.info(
            "Computing SBERT cosine [%d train × %d val] …",
            len(train_sbert), len(val_sbert),
        )
        sim   : np.ndarray = cosine_similarity(train_sbert, val_sbert)
        msims : np.ndarray = sim.max(axis=0)      # [n_val]
        bidx  : np.ndarray = sim.argmax(axis=0)   # [n_val]

        stats = self._stats(msims)
        report["sim_stats"] = stats
        logger.info(
            "Sim — mean=%.4f  median=%.4f  max=%.4f",
            stats["mean"], stats["median"], stats["max"],
        )

        # C — top-K pairs
        pairs = self._top_pairs(train_df, val_df, msims, bidx)
        report["top_pairs"] = pairs
        for rank, p in enumerate(pairs, 1):
            flag = "⚠ SAME ANS" if p["same_answer"] else "diff ans"
            logger.info(
                "#%02d cos=%.4f %s | val: %s | train: %s",
                rank, p["sim"], flag,
                p["val_q"][:55], p["train_q"][:55],
            )

        # D — near-perfect pairs
        ri, ci = np.where(sim >= 0.9999)
        report["perfect_count"] = int(len(ri))
        logger.info("Pairs with cosine ≥ 0.9999: %d", len(ri))

        # E — expose max_sims for reuse in pipeline (avoids recomputation)
        report["max_sims"] = msims

        if wandb_run is not None:
            wandb_run.log({
                "audit/exact_fp_overlap"  : report["exact_fp"],
                "audit/option_fp_overlap" : report["option_fp"],
                "audit/sbert_sim_mean"    : stats["mean"],
                "audit/sbert_sim_max"     : stats["max"],
                "audit/n_above_0.90"      : stats.get(">0.90", 0),
                "audit/perfect_pairs"     : int(len(ri)),
            })

        return report

    # ──────────────────────────────────────────────────────────────────────

    def _stats(self, sims: np.ndarray) -> dict:
        s = dict(
            mean   = float(np.mean(sims)),
            median = float(np.median(sims)),
            std    = float(np.std(sims)),
            max    = float(np.max(sims)),
            min    = float(np.min(sims)),
        )
        for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
            s[f">{t:.2f}"] = int((sims >= t).sum())
        return s

    def _top_pairs(
        self,
        train_df : pd.DataFrame,
        val_df   : pd.DataFrame,
        msims    : np.ndarray,
        bidx     : np.ndarray,
    ) -> list:
        top_val_indices = np.argsort(msims)[::-1][: self.top_k]
        top_train_indices = bidx[top_val_indices]

        # Fetch rows in bulk to avoid repeated iloc overhead
        val_rows   = val_df.iloc[top_val_indices].reset_index(drop=True)
        train_rows = train_df.iloc[top_train_indices].reset_index(drop=True)

        out = []
        for rank_i in range(len(top_val_indices)):
            vi = top_val_indices[rank_i]
            vr = val_rows.iloc[rank_i]
            tr = train_rows.iloc[rank_i]
            out.append(dict(
                sim         = float(msims[vi]),
                val_id      = vr.get("id", vi),
                train_id    = tr.get("id", top_train_indices[rank_i]),
                val_q       = normalize_text(
                                  str(vr.get("prompt", "")))[:120],
                train_q     = normalize_text(
                                  str(tr.get("prompt", "")))[:120],
                same_answer = (
                    str(vr.get("answer", "?")) ==
                    str(tr.get("answer", "?"))
                ),
            ))
        return out