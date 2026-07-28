# scripts/run_training.py

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Make src importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import Config
from src.data.data_loader import DataLoader as MCQDataLoader
from src.preprocessing import preprocessing as Preprocessor
from src.pipeline.training_pipeline import TrainingPipeline
from src.ensemble.fuser import ScoreFuser as Fuser
from src.evaluation.evaluator import MAPAtKEvaluator as Evaluator

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger("run_training")


def run_milestone4() -> Dict[str, Any]:


    # ── 1. Configuration ───────────────────────────────────────────
    cfg = Config()
    logger.info(f"Device: {cfg.device} | GPUs: {cfg.n_gpus}")

    # ── 2. Load data ───────────────────────────────────────────────
    loader = MCQDataLoader(cfg)
    train_df_raw = loader.load_train()
    test_df_raw = loader.load_test()

    # ── 3. Preprocess ──────────────────────────────────────────────
    preprocessor = Preprocessor(cfg)
    train_df = preprocessor.fit_transform(train_df_raw)
    test_df = preprocessor.transform(test_df_raw)

    # ── 4. Train / val split (reuse Milestone 3 split) ────────────
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=train_df[cfg.answer_col],
    )
    logger.info(
        f"Split — train: {len(train_df)} | "
        f"val: {len(val_df)} | test: {len(test_df)}"
    )

    # ── 5. RAG contexts (reuse Milestone 3 retrieval) ─────────────
    # If RAG pipeline is not available, default to empty strings
    try:
        try:
            from src.retrieval.rag_pipeline import RAGPipeline
        except ImportError:
            from src.rag.rag_pipeline import RAGPipeline

        rag = RAGPipeline(cfg)
        if hasattr(rag, "fit"):
            rag.fit(train_df)
        elif hasattr(rag, "build_index"):
            rag.build_index(train_df)

        if hasattr(rag, "retrieve_batch"):
            train_ctx = rag.retrieve_batch(train_df)
            val_ctx = rag.retrieve_batch(val_df)
            test_ctx = rag.retrieve_batch(test_df)
        elif hasattr(rag, "get_retrieval_context"):
            train_ctx = rag.get_retrieval_context(train_df)
            val_ctx = rag.get_retrieval_context(val_df)
            test_ctx = rag.get_retrieval_context(test_df)
        else:
            raise AttributeError("RAGPipeline has no supported context method.")

        logger.info("RAG contexts loaded.")
    except Exception as exc:
        logger.warning(f"RAG unavailable ({exc}), using empty contexts.")
        train_ctx = [""] * len(train_df)
        val_ctx = [""] * len(val_df)
        test_ctx = [""] * len(test_df)

    # ── 6. Run training pipeline ───────────────────────────────────
    pipeline = TrainingPipeline(cfg)
    results = pipeline.run(
        train_df,
        val_df,
        test_df,
        train_ctx,
        val_ctx,
        test_ctx,
    )

    # ── 7. Ensemble + evaluation ──────────────────────────────────
    evaluator = Evaluator(cfg)
    fuser = Fuser(cfg)

    # Fuse all test logits
    fused_test_logits = fuser.fuse(results["all_test_logits"])
    test_preds = evaluator.scores_to_top_k_predictions(
        fused_test_logits, cfg.options
    )

    # Fuse val logits for final MAP@3 report
    fused_val_logits = fuser.fuse(results["all_val_logits"])
    val_preds = evaluator.scores_to_top_k_predictions(
        fused_val_logits, cfg.options
    )

    final_metrics = evaluator.evaluate(
        val_df, val_preds, cfg, split="ensemble_val"
    )
    logger.info(
        f"Final ensemble MAP@3: "
        f"{final_metrics.get('ensemble_val/map@3', 0):.4f}"
    )

    # ── 8. Package outputs like Milestone 3 ────────────────────────
    results.setdefault("metrics", {})
    results["metrics"].update(final_metrics)

    results["train_df"] = train_df
    results["val_df"] = val_df
    results["test_df"] = test_df

    results["train_contexts"] = train_ctx
    results["val_contexts"] = val_ctx
    results["test_contexts"] = test_ctx

    results["fused_val_logits"] = fused_val_logits
    results["fused_test_logits"] = fused_test_logits

    results["val_preds"] = val_preds
    results["test_preds"] = test_preds

    # Optional preview
    preview = pd.DataFrame(
        {
            cfg.id_col: test_df[cfg.id_col].astype(str).head(10).tolist(),
            "prediction": [
                " ".join(p[: cfg.top_k]) for p in test_preds[:10]
            ],
        }
    )
    logger.info(f"\n{preview.to_string(index=False)}")

    return results


def main() -> Dict[str, Any]:
    return run_milestone4()


if __name__ == "__main__":
    outputs = main()
    logger.info(f"Milestone 4 completed. Output keys: {list(outputs.keys())}")