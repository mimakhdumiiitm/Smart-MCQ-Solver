# cell 5
# src/ranker.py
from typing import List, Dict, Optional

from config.config import (
    STRATEGY,
    ENSEMBLE_WEIGHTS,
    TOP_K)


class EnsembleRanker:
    def __init__(
        self,
        strategy        : str  = STRATEGY,
        ensemble_weights: dict = None,
        top_k           : int  = TOP_K,
    ):
        self.strategy         = strategy
        self.ensemble_weights = ensemble_weights or ENSEMBLE_WEIGHTS
        self.top_k            = top_k

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
        if not score_dict:
            return score_dict

        vals = list(score_dict.values())
        mn   = min(vals)
        mx   = max(vals)
        rng  = mx - mn if mx != mn else 1.0

        return {k: (v - mn) / rng for k, v in score_dict.items()}

    # ------------------------------------------------------------------
    # COMBINE SCORES
    # ------------------------------------------------------------------

    def combine(
        self,
        emb_scores : Dict[str, float],
        zs_scores  : Dict[str, float],
    ) -> Dict[str, float]:
        emb_norm  = self.normalize(emb_scores)
        zs_norm   = self.normalize(zs_scores)
        w_emb     = self.ensemble_weights.get("embedding", 0.5)
        w_zs      = self.ensemble_weights.get("zeroshot",  0.5)
        all_labels = set(emb_norm) | set(zs_norm)

        combined = {}
        for label in all_labels:
            e = emb_norm.get(label, 0.0)
            z = zs_norm.get(label, 0.0)
            combined[label] = w_emb * e + w_zs * z

        return combined

    # ------------------------------------------------------------------
    # RANK
    # ------------------------------------------------------------------

    @staticmethod
    def rank(score_dict: Dict[str, float]) -> List[str]:
        return sorted(score_dict, key=score_dict.get, reverse=True)

    # ------------------------------------------------------------------
    # PREDICT ONE
    # ------------------------------------------------------------------

    def predict_one(
        self,
        emb_scores : Dict[str, float],
        zs_scores  : Dict[str, float],
        strategy   : Optional[str] = None,
    ) -> List[str]:
        strat = strategy or self.strategy

        if strat == "embedding":
            final_scores = emb_scores
        elif strat == "zeroshot":
            final_scores = zs_scores
        elif strat == "ensemble":
            final_scores = self.combine(emb_scores, zs_scores)
        else:
            raise ValueError(
                f"Unknown strategy: '{strat}'. "
                f"Choose from: 'embedding', 'zeroshot', 'ensemble'."
            )

        return self.rank(final_scores)[: self.top_k]

    # ------------------------------------------------------------------
    # PREDICT ALL
    # ------------------------------------------------------------------

    def predict_all(
        self,
        records        : List[dict],
        emb_scores_list: List[Dict[str, float]],
        zs_scores_list : List[Dict[str, float]],
    ) -> List[dict]:
        results = []
        for rec, emb_s, zs_s in zip(records, emb_scores_list, zs_scores_list):
            top_k    = self.predict_one(emb_s, zs_s)
            combined = self.combine(emb_s, zs_s)

            results.append({
                "id"             : rec["id"],
                "prediction"     : " ".join(top_k),
                "top_labels"     : top_k,
                "answer"         : rec.get("answer"),
                "emb_scores"     : emb_s,
                "zs_scores"      : zs_s,
                "combined_scores": combined,
            })
        return results