# src/DeBERTa/auditor.py
"""
Leakage auditor — uses SBERT embeddings instead of BoW.
Pre-computed SBERT matrix passed in (already done during dedup).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.DeBERTa.data import normalize_text

logger = logging.getLogger("DeBERTa.Auditor")


class LeakageAuditor:
    """
    Post-split leakage audit using pre-computed SBERT vectors.

    Parameters
    ──────────
    top_k : number of most-similar pairs to report
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def run(
        self,
        train_df    : pd.DataFrame,
        val_df      : pd.DataFrame,
        train_sbert : np.ndarray,    # [N_train, 384] L2-normed
        val_sbert   : np.ndarray,    # [N_val,   384] L2-normed
        wandb_run   = None,
    ) -> dict:

        logger.info("─" * 60)
        logger.info("LEAKAGE AUDIT  (SBERT cosine)")
        logger.info("─" * 60)
        report = {}

        # A — exact fingerprint overlap
        for col, label in [('exact_fp',  'question+options'),
                           ('option_fp', 'option-set only')]:
            overlap = (
                set(train_df.get(col, pd.Series(dtype=str))) &
                set(val_df.get(col,   pd.Series(dtype=str)))
            )
            report[col] = len(overlap)
            logger.info(f"Exact {label} overlap: {len(overlap):,}")

        # B — SBERT cosine similarity
        logger.info(
            f"Computing SBERT cosine "
            f"[{len(train_sbert)} train × {len(val_sbert)} val] …"
        )
        sim   = cosine_similarity(train_sbert, val_sbert)
        msims = sim.max(axis=0)
        bidx  = sim.argmax(axis=0)

        stats = self._stats(msims)
        report['sim_stats'] = stats
        logger.info(
            f"Sim — mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} "
            f"max={stats['max']:.4f}"
        )
        for k, v in stats.items():
            if k.startswith('>'):
                logger.info(f"  {k} → {v:,} pairs")

        # C — top-K most similar pairs
        pairs = self._top_pairs(train_df, val_df, msims, bidx)
        report['top_pairs'] = pairs
        for rank, p in enumerate(pairs, 1):
            flag = "⚠ SAME ANS" if p['same_answer'] else "diff ans"
            logger.info(
                f"#{rank:02d} cos={p['sim']:.4f} {flag} | "
                f"val: {p['val_q'][:55]} | "
                f"train: {p['train_q'][:55]}"
            )

        # D — near-perfect pairs (≥ 0.9999)
        ri, ci = np.where(sim >= 0.9999)
        report['perfect_count'] = len(ri)
        logger.info(f"Pairs with cosine ≥ 0.9999: {len(ri):,}")
        for r, c in zip(ri[:10], ci[:10]):
            tq = normalize_text(str(train_df.iloc[r].get('prompt', '')),
                                lowercase=True)
            vq = normalize_text(str(val_df.iloc[c].get('prompt', '')),
                                lowercase=True)
            reason = 'EXACT' if tq == vq else 'NEAR_PARAPHRASE'
            logger.info(f"  [{reason}] "
                        f"train: {tq[:60]} | val: {vq[:60]}")

        # W&B
        if wandb_run is not None:
            wandb_run.log({
                "audit/exact_fp_overlap"  : report['exact_fp'],
                "audit/option_fp_overlap" : report['option_fp'],
                "audit/sbert_sim_mean"    : stats['mean'],
                "audit/sbert_sim_max"     : stats['max'],
                "audit/n_above_0.90"      : stats.get('>0.90', 0),
                "audit/perfect_pairs"     : len(ri),
            })

        return report

    # ── helpers ───────────────────────────────────────────────────────────────

    def _stats(self, sims: np.ndarray) -> dict:
        s = dict(
            mean   = float(np.mean(sims)),
            median = float(np.median(sims)),
            std    = float(np.std(sims)),
            max    = float(np.max(sims)),
            min    = float(np.min(sims)),
        )
        for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
            s[f'>{t:.2f}'] = int((sims >= t).sum())
        return s

    def _top_pairs(self, train_df, val_df, msims, bidx) -> list:
        top = np.argsort(msims)[::-1][: self.top_k]
        out = []
        for vi in top:
            ti = bidx[vi]
            vr = val_df.iloc[vi]
            tr = train_df.iloc[ti]
            out.append(dict(
                sim         = float(msims[vi]),
                val_id      = vr.get('id', vi),
                train_id    = tr.get('id', ti),
                val_q       = normalize_text(
                    str(vr.get('prompt', '')))[:120],
                train_q     = normalize_text(
                    str(tr.get('prompt', '')))[:120],
                same_answer = (
                    str(vr.get('answer', '?')) ==
                    str(tr.get('answer', '?'))
                ),
            ))
        return out