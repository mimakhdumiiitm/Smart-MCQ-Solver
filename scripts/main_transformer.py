# cell 6
# scripts/main_transformer.py
import os
import time

import wandb
from utils.gpu_utils import setup_gpu_environment

DEVICE = setup_gpu_environment()

# ------------------------------------------------------------------
# CONFIG  (after device is detected)
# ------------------------------------------------------------------

from config.config import (
    # Paths
    TRAIN_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
    SUBMISSION_OUT_PATH,
    RESULTS_PLOT_PATH,
    # Column names
    OPTION_COLS,
    TOP_K,
    # Transformer settings
    EMBEDDING_MODEL,
    ZEROSHOT_MODEL,
    TRANSFORMER_BATCH_SIZE,
    STRATEGY,
    ENSEMBLE_WEIGHTS,
)

# ------------------------------------------------------------------
# MODULE IMPORTS
# ------------------------------------------------------------------

from utils.wandb_utils          import setup_wandb
from utils.data_loader          import TransformerDataLoader
from utils.metrics              import MAP3Evaluator
from utils.submission           import SubmissionGenerator
from src.transformer_embeddings import (
    EmbeddingScorer,
    ZeroShotScorer,
    smoke_test,
)
from src.ranker import EnsembleRanker

print("All modules imported successfully.")

# ==================================================================
# STEP 1: W&B INITIALIZATION
# ==================================================================

def step_init_wandb() -> wandb.run:
    print("\n" + "=" * 55)
    print("STEP 1 : W&B INITIALIZATION")
    print("=" * 55)

    run = setup_wandb()
    return run


# ==================================================================
# STEP 2: DATA LOADING
# ==================================================================

def step_load_data():
    print("\n" + "=" * 55)
    print("STEP 2 : DATA LOADING")
    print("=" * 55)

    loader = TransformerDataLoader(
        train_path    = TRAIN_PROCESSED_PATH,
        test_path     = TEST_PROCESSED_PATH,
        option_cols   = OPTION_COLS,
        option_labels = OPTION_COLS,
    )

    train_df   = loader.load_train()
    test_df    = loader.load_test()
    train_data = loader.format_rows(train_df)
    test_data  = loader.format_rows(test_df)

    print(f"\nData ready | Train: {len(train_data)} | Test: {len(test_data)}")

    # Print one sample record for verification
    sample = train_data[0]
    print(f"\nSample record:")
    print(f"   Prompt  : {sample['prompt'][:80]}...")
    print(f"   Options : {sample['options']}")
    print(f"   Answer  : {sample['answer']}")

    return train_data, test_data


# ==================================================================
# STEP 3: MODEL LOADING
# ==================================================================

def step_load_models():
    print("\n" + "=" * 55)
    print("STEP 3 : MODEL LOADING")
    print("=" * 55)

    emb_scorer = EmbeddingScorer(
        model_name = EMBEDDING_MODEL,
        device     = DEVICE,
        batch_size = TRANSFORMER_BATCH_SIZE,
    )

    zs_scorer = ZeroShotScorer(
        model_name = ZEROSHOT_MODEL,
        device     = DEVICE,
        batch_size = TRANSFORMER_BATCH_SIZE,
    )

    return emb_scorer, zs_scorer


# ==================================================================
# STEP 4: SMOKE TEST
# ==================================================================

def step_smoke_test(emb_scorer: EmbeddingScorer,
                    zs_scorer: ZeroShotScorer) -> None:
    print("\n" + "=" * 55)
    print("STEP 4 : SMOKE TEST")
    print("=" * 55)

    ok = smoke_test(emb_scorer, zs_scorer)
    if not ok:
        raise RuntimeError(
            "Smoke test failed. Check model loading output above."
        )


# ==================================================================
# STEP 5: SCORING
# ==================================================================

def step_score(train_data: list,
               test_data: list,
               emb_scorer: EmbeddingScorer,
               zs_scorer: ZeroShotScorer):
    print("\n" + "=" * 55)
    print("STEP 5 : SCORING")
    print("=" * 55)

    print("\n[1/4] Embedding scoring — TRAIN...")
    train_emb_scores = emb_scorer.score_batch(train_data)

    print("\n[2/4] Zero-shot scoring — TRAIN...")
    train_zs_scores = zs_scorer.score_batch(train_data)

    print("\n[3/4] Embedding scoring — TEST...")
    test_emb_scores = emb_scorer.score_batch(test_data)

    print("\n[4/4] Zero-shot scoring — TEST...")
    test_zs_scores = zs_scorer.score_batch(test_data)

    return train_emb_scores, train_zs_scores, test_emb_scores, test_zs_scores


# ==================================================================
# STEP 6: PREDICTION
# ==================================================================

def step_predict(train_data: list,
                 test_data: list,
                 train_emb_scores: list,
                 train_zs_scores: list,
                 test_emb_scores: list,
                 test_zs_scores: list):
    print("\n" + "=" * 55)
    print("STEP 6 : GENERATING PREDICTIONS")
    print("=" * 55)

    ranker = EnsembleRanker(
        strategy         = STRATEGY,
        ensemble_weights = ENSEMBLE_WEIGHTS,
        top_k            = TOP_K,
    )

    train_results = ranker.predict_all(
        train_data, train_emb_scores, train_zs_scores
    )
    test_results = ranker.predict_all(
        test_data, test_emb_scores, test_zs_scores
    )

    print(f"Train predictions : {len(train_results)}")
    print(f"Test  predictions : {len(test_results)}")

    return ranker, train_results, test_results


# ==================================================================
# STEP 7: EVALUATION
# ==================================================================

def step_evaluate(train_data: list,
                  train_results: list,
                  train_emb_scores: list,
                  train_zs_scores: list,
                  ranker: EnsembleRanker,
                  pipeline_time: float):
    print("\n" + "=" * 55)
    print("STEP 7 : MAP@3 EVALUATION")
    print("=" * 55)

    evaluator     = MAP3Evaluator(top_k=TOP_K)
    train_metrics = evaluator.compute_map3(train_results)

    print(f"\nTotal pipeline time : {pipeline_time:.1f}s")
    print("\nRESULTS:")
    for k, v in train_metrics.items():
        print(f"   {k:<25}: {v}")

    # Log core metrics and timing to W&B
    wandb.log({
        **{f"train/{k}": v for k, v in train_metrics.items()},
        "pipeline_time_seconds": pipeline_time,
        "device"               : DEVICE,
    })

    # Strategy comparison
    print("\nComparing strategies...")
    strategy_df = evaluator.compare_strategies(
        train_data, train_emb_scores, train_zs_scores, ranker
    )

    print("\n" + "=" * 55)
    print("STRATEGY COMPARISON")
    print("=" * 55)
    print(strategy_df[
        ["Strategy", "MAP@3", "hit@1", "hit@2", "hit@3"]
    ].to_string(index=False))
    print("=" * 55)
    print(f"\nBest MAP@3 (Ensemble): {train_metrics['MAP@3']}")

    # Regenerate clean results for plotting
    # (compare_strategies mutates top_labels internally)
    fresh_train_results = ranker.predict_all(
        train_data, train_emb_scores, train_zs_scores
    )

    evaluator.plot_and_log(
        metrics     = train_metrics,
        strategy_df = strategy_df,
        results     = fresh_train_results,
        plot_path   = RESULTS_PLOT_PATH,
    )

    return evaluator, train_metrics, strategy_df


# ==================================================================
# STEP 8: SUBMISSION
# ==================================================================

def step_submission(test_results: list) -> None:
    print("\n" + "=" * 55)
    print("STEP 8 : GENERATE SUBMISSION")
    print("=" * 55)

    sub_gen = SubmissionGenerator(
        option_labels = OPTION_COLS,
        top_k         = TOP_K,
        output_path   = SUBMISSION_OUT_PATH,
    )

    sub_df = sub_gen.generate(test_results)
    sub_gen.validate(sub_df)

    print("\nSubmission Preview:")
    print(sub_df.head(10).to_string(index=False))

    sub_gen.save(sub_df)
