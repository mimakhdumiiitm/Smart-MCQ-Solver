# cell 4
# src/transformer_embeddings.py
import time
import numpy as np
import torch
from typing import List, Dict

from config.config import (
    EMBEDDING_MODEL,
    ZEROSHOT_MODEL,
    TRANSFORMER_BATCH_SIZE,
    MAX_SEQ_LENGTH,
    TOP_K,
)
from .model_loader import safe_load_sbert, safe_load_nli


# ==================================================================
# EMBEDDING SCORER  (SBERT)
# ==================================================================

class EmbeddingScorer:
    def __init__(
        self,
        model_name : str = EMBEDDING_MODEL,
        device     : str = "cuda",
        batch_size : int = TRANSFORMER_BATCH_SIZE,
    ):
        self.batch_size = batch_size

        # Load with safety wrapper (CPU fallback built in)
        self.model = safe_load_sbert(model_name, device)

        # Detect actual device the model landed on
        self.device = str(
            next(self.model._modules["0"].auto_model.parameters()).device
        )
        print(f"   EmbeddingScorer actual device: {self.device}")

    # ------------------------------------------------------------------
    # ENCODE
    # ------------------------------------------------------------------

    def encode(
        self,
        texts      : List[str],
        batch_size : int = None,
    ) -> np.ndarray:
        bs = batch_size or self.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size           = bs,
            show_progress_bar    = False,
            convert_to_numpy     = True,
            normalize_embeddings = True,   # L2 normalize → dot == cosine
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # BATCH SCORING
    # ------------------------------------------------------------------

    def score_batch(self, records: List[dict]) -> List[Dict[str, float]]:
        print(f"SBERT scoring {len(records)} records...")
        t0 = time.time()

        # Collect all texts for batch encoding
        all_prompts = [rec["prompt"] for rec in records]
        all_options = []   # (record_idx, label, text)
        for idx, rec in enumerate(records):
            for label, text in rec["options"].items():
                all_options.append((idx, label, text))

        # Encode prompts and options in separate batches
        # (more memory efficient than interleaving)
        prompt_embs = self.encode(all_prompts)
        option_embs = self.encode([o[2] for o in all_options])

        # Compute dot products (L2-normalized → cosine similarity)
        scores = [{} for _ in records]
        for (rec_idx, label, _), opt_emb in zip(all_options, option_embs):
            sim = float(np.dot(prompt_embs[rec_idx], opt_emb))
            scores[rec_idx][label] = sim

        print(f"SBERT done in {time.time() - t0:.1f}s")
        return scores


# ==================================================================
# ZERO-SHOT SCORER  (NLI)
# ==================================================================

class ZeroShotScorer:
    def __init__(
        self,
        model_name : str = ZEROSHOT_MODEL,
        device     : str = "cuda",
        batch_size : int = TRANSFORMER_BATCH_SIZE,
    ):
        self.batch_size = batch_size

        # Load with safety wrapper (CPU fallback built in)
        self.tokenizer, self.model, self.actual_device = safe_load_nli(
            model_name, device
        )
        self.device = torch.device(self.actual_device)

        # Identify entailment label index from model config
        self.entail_idx = self._find_entail_idx()

    # ------------------------------------------------------------------
    # ENTAILMENT INDEX DETECTION
    # ------------------------------------------------------------------

    def _find_entail_idx(self) -> int:
        id2label = self.model.config.id2label
        print(f"   NLI labels: {id2label}")
        for idx, label in id2label.items():
            if "entail" in label.lower():
                print(f"   Entailment index: {idx}")
                return idx
        print("   'entailment' not found in labels, using index 0")
        return 0

    # ------------------------------------------------------------------
    # HYPOTHESIS BUILDER
    # ------------------------------------------------------------------

    @staticmethod
    def build_hypothesis(option_text: str) -> str:
        return f"The correct answer is: {option_text}"

    # ------------------------------------------------------------------
    # SCORE PAIRS
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_pairs(
        self,
        premises   : List[str],
        hypotheses : List[str],
    ) -> np.ndarray:
        encoded = self.tokenizer(
            premises,
            hypotheses,
            padding        = True,
            truncation     = True,
            max_length     = MAX_SEQ_LENGTH,
            return_tensors = "pt",
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Forward pass — disable autocast for P100 float32 safety
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.model(**encoded).logits

        probs  = torch.softmax(logits.float(), dim=-1)
        entail = probs[:, self.entail_idx].cpu().numpy()
        return entail.astype(np.float32)

    # ------------------------------------------------------------------
    # BATCH SCORING
    # ------------------------------------------------------------------

    def score_batch(self, records: List[dict]) -> List[Dict[str, float]]:
        print(f"Zero-shot scoring {len(records)} records...")
        t0 = time.time()

        # Flatten all (premise, hypothesis) pairs with metadata
        premises   = []
        hypotheses = []
        meta       = []   # (record_idx, label)

        for idx, rec in enumerate(records):
            for label, text in rec["options"].items():
                premises.append(rec["prompt"])
                hypotheses.append(self.build_hypothesis(text))
                meta.append((idx, label))

        scores = [{} for _ in records]
        total  = len(premises)
        bs     = self.batch_size

        for start in range(0, total, bs):
            end   = min(start + bs, total)
            probs = self._score_pairs(premises[start:end],
                                      hypotheses[start:end])

            for i, prob in enumerate(probs):
                rec_idx, label = meta[start + i]
                scores[rec_idx][label] = float(prob)

            # Progress reporting every 10 batches
            if (start // bs) % 10 == 0:
                pct = (end / total) * 100
                print(f"   Progress: {end}/{total} ({pct:.0f}%)")

        print(f"Zero-shot done in {time.time() - t0:.1f}s")
        return scores


# ==================================================================
# SMOKE TEST
# ==================================================================

def smoke_test(
    emb_scorer : EmbeddingScorer,
    zs_scorer  : ZeroShotScorer,
) -> bool:
    print("Running smoke test...")

    test_records = [{
        "id"    : 999,
        "prompt": "What is the capital of France?",
        "options": {
            "A": "London",
            "B": "Berlin",
            "C": "Paris",
            "D": "Madrid",
            "E": "Rome",
        },
        "answer": "C",
    }]

    # Test EmbeddingScorer
    try:
        emb_scores = emb_scorer.score_batch(test_records)
        best_emb   = max(emb_scores[0], key=emb_scores[0].get)
        print(f"Embedding scores : {emb_scores[0]}")
        print(f"   Best (embedding) : {best_emb} "
              f"{'✓' if best_emb == 'C' else '✗'}")
    except Exception as e:
        print(f"Embedding scorer error: {e}")
        return False

    # Test ZeroShotScorer
    try:
        zs_scores = zs_scorer.score_batch(test_records)
        best_zs   = max(zs_scores[0], key=zs_scores[0].get)
        print(f"Zero-shot scores : {zs_scores[0]}")
        print(f"   Best (zero-shot) : {best_zs} "
              f"{'✓' if best_zs == 'C' else '✗'}")
    except Exception as e:
        print(f"Zero-shot scorer error: {e}")
        return False

    print("Smoke test passed! Safe to run full pipeline.")
    return True