# auditor.py
import logging
import numpy as np
import pandas as pd

from data import normalize_text
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("Auditor")


class LeakageAuditor:
    """
    Post-split leakage audit using BoW vectors + our own cosine.
    No external similarity libraries used.
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def run(self, train_df, val_df,
            train_bow, val_bow,       # L2-normed BoW float32 arrays
            wandb_run=None) -> dict:

        logger.info("─" * 60)
        logger.info("LEAKAGE AUDIT")
        logger.info("─" * 60)
        report = {}

        # A — exact fingerprint overlap
        for col, label in [('exact_fp',  'question+options'),
                           ('option_fp', 'option-set only')]:
            overlap = set(train_df.get(col, pd.Series(dtype=str))) & \
                      set(val_df.get(col, pd.Series(dtype=str)))
            report[col] = len(overlap)
            logger.info(f"Exact {label} overlap: {len(overlap):,}")

        # B — BoW cosine similarity (our own implementation)
        logger.info(f"Computing cosine similarity "
                    f"[{len(train_bow)} train × {len(val_bow)} val] …")

        sim = cosine_similarity(train_bow, val_bow)
        msims = sim.max(axis=0)                               
        bidx  = sim.argmax(axis=0)

        stats = self._stats(msims)
        report['sim_stats'] = stats
        logger.info(f"Sim — mean={stats['mean']:.4f} "
                    f"median={stats['median']:.4f} "
                    f"max={stats['max']:.4f}")
        for k, v in stats.items():
            if k.startswith('>'):
                logger.info(f"  {k} → {v:,} pairs")

        # C — top-K pairs
        pairs = self._top_pairs(train_df, val_df, msims, bidx)
        report['top_pairs'] = pairs
        for rank, p in enumerate(pairs, 1):
            flag = "⚠ SAME ANS" if p['same_answer'] else "diff ans"
            logger.info(f"#{rank:02d} cos={p['sim']:.4f} {flag} | "
                        f"val: {p['val_q'][:55]} | "
                        f"train: {p['train_q'][:55]}")

        # D — near-perfect pairs (≥ 0.9999)
        ri, ci = np.where(sim >= 0.9999)
        report['perfect_count'] = len(ri)
        logger.info(f"Pairs with cosine ≥ 0.9999: {len(ri):,}")
        for r, c in zip(ri[:10], ci[:10]):
            tq = normalize_text(str(train_df.iloc[r].get('prompt', '')))
            vq = normalize_text(str(val_df.iloc[c].get('prompt', '')))
            reason = 'EXACT' if tq == vq else 'NEAR_PARAPHRASE'
            logger.info(f"  [{reason}] "
                        f"train: {tq[:60]} | val: {vq[:60]}")

        # W&B
        if wandb_run is not None:
            wandb_run.log({
                "audit/exact_fp_overlap"  : report['exact_fp'],
                "audit/option_fp_overlap" : report['option_fp'],
                "audit/sim_mean"          : stats['mean'],
                "audit/sim_max"           : stats['max'],
                "audit/n_above_0.90"      : stats.get('>0.90', 0),
                "audit/perfect_pairs"     : len(ri),
            })

        return report

    # ── helpers ──────────────────────────────────────────────────────────────

    def _stats(self, sims: np.ndarray) -> dict:
        s = dict(mean   = float(np.mean(sims)),
                 median = float(np.median(sims)),
                 std    = float(np.std(sims)),
                 max    = float(np.max(sims)),
                 min    = float(np.min(sims)))
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
                val_q       = normalize_text(str(vr.get('prompt', '')))[:120],
                train_q     = normalize_text(str(tr.get('prompt', '')))[:120],
                val_answer  = str(vr.get('answer', '?')),
                train_answer= str(tr.get('answer', '?')),
                same_answer = (str(vr.get('answer', '?')) ==
                               str(tr.get('answer', '?'))),
            ))
        return out