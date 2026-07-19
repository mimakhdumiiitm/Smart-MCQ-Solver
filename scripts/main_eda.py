# main.py

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config.config import Config, set_seed, get_logger

from utils.wandb_init import (
    init_wandb,
    log_model_metrics,
    finish_run,
)
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import TextPreprocessor
from src.evaluation.evaluator import MAPAtKEvaluator
from src.models.tfidf_ranker import TFIDFRanker
from src.models.w2v_ranker import Word2VecRanker
from src.models.sbert_ranker import SBERTRanker
from src.ensemble.fuser import ScoreFuser
from utils.submission import SubmissionGenerator

from src.visualizaton.visualization import (
    plot_answer_distribution,
    plot_text_length_distributions,
    plot_top_words,
    plot_wordcloud,
    plot_similarity_distributions,
    plot_map_comparison,
    plot_rank_distribution,
    plot_w2v_pca,
)

logger = get_logger("Main")


# ══════════════════════════════════════════════════════════════════════
# PHASE 0 — Config, seed, W&B
# ══════════════════════════════════════════════════════════════════════

def setup(
    run_name: str = "phase1-baseline",
    use_wandb_override: Optional[bool] = None,
) -> Tuple[Config, Dict[str, Optional[object]], MAPAtKEvaluator]:
    cfg = Config()
    set_seed(cfg.seed)

    if use_wandb_override is not None:
        cfg.use_wandb = use_wandb_override

    # ── Create one run per model so all three appear in W&B ──────────────────
    wandb_runs = {
        "tfidf"   : init_wandb(cfg, run_name="phase1-tfidf",    model_tag="tfidf"),
        "word2vec": init_wandb(cfg, run_name="phase1-word2vec",  model_tag="word2vec"),
        "sbert"   : init_wandb(cfg, run_name="phase1-sbert",     model_tag="sbert"),
    }

    evaluator = MAPAtKEvaluator(k=cfg.top_k)
    MAPAtKEvaluator.run_unit_tests()

    logger.info("Setup complete.")
    return cfg, wandb_runs, evaluator


# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — Data loading, preprocessing, EDA visuals, train/val split
# ══════════════════════════════════════════════════════════════════════

def load_and_preprocess(
    cfg           : Config,
    run_eda_plots : bool = True,
) -> Tuple[
    "pd.DataFrame",   # full processed train
    "pd.DataFrame",   # full processed test
    "pd.DataFrame",   # fit split  (80 %)
    "pd.DataFrame",   # val split  (20 %)
    List[str],        # option column labels  e.g. ["A","B","C","D","E"]
]:

    # ── Raw data ────────────────────────────────────────────────
    loader       = DataLoader(cfg)
    raw_train_df = loader.load_train()
    raw_test_df  = loader.load_test()
    logger.info(f"Raw train: {raw_train_df.shape}  |  Raw test: {raw_test_df.shape}")

    # ── Preprocessing (cached) ──────────────────────────────────
    prep     = TextPreprocessor()
    train_df = prep.process_dataframe(raw_train_df, cfg, split="train")
    test_df  = prep.process_dataframe(raw_test_df,  cfg, split="test")

    # ── EDA visualisations ──────────────────────────────────────
    if run_eda_plots:
        _run_eda_plots(train_df, cfg)

    # ── Stratified train / val split ────────────────────────────
    stratify_col = (
        train_df[cfg.answer_col]
        if cfg.answer_col in train_df.columns
        else np.zeros(len(train_df))
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    train_idx, val_idx = next(skf.split(train_df, stratify_col))

    fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    logger.info(f"Fit: {len(fit_df)}  |  Val: {len(val_df)}")

    option_cols = [c for c in cfg.options if c in train_df.columns]

    return train_df, test_df, fit_df, val_df, option_cols


def _run_eda_plots(train_df: "pd.DataFrame", cfg: Config) -> None:
    # Answer distribution
    if cfg.answer_col in train_df.columns:
        plot_answer_distribution(train_df, answer_col=cfg.answer_col)

    # Text length histograms
    plot_text_length_distributions(
        train_df, cols=[cfg.prompt_col] + cfg.options
    )

    # Top words + word cloud (prompt column)
    all_tokens = [
        tok
        for text in train_df["prompt_clean"].fillna("")
        for tok in text.split()
    ]
    word_freq = Counter(all_tokens)
    plot_top_words(
        word_freq, title="Top Prompt Words",
        n=25, filename="top_prompt_words.png",
    )
    plot_wordcloud(
        all_tokens, title="Prompt Word Cloud",
        filename="prompt_wordcloud.png",
    )


# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — Train / load all three rankers, evaluate individually
# ══════════════════════════════════════════════════════════════════════

def train_models(
    cfg          : Config,
    fit_df       : "pd.DataFrame",
    val_df       : "pd.DataFrame",
    test_df      : "pd.DataFrame",
    option_cols  : List[str],
    evaluator    : MAPAtKEvaluator,
    wandb_runs   : Optional[Dict[str, Optional[object]]] = None,  # ← Dict now
    run_sim_plots: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    """
    Same as before — wandb_runs is now a dict keyed by model name.
    Each model logs to its own W&B run.
    """
    # keep None safe
    if wandb_runs is None:
        wandb_runs = {"tfidf": None, "word2vec": None, "sbert": None}

    models : Dict[str, Any]                   = {}
    scores : Dict[str, Dict[str, np.ndarray]] = {"val": {}, "test": {}}
    metrics: Dict[str, Dict[str, float]]      = {}

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    models["tfidf"], scores, metrics["tfidf"] = _fit_tfidf(
        cfg, fit_df, val_df, test_df, option_cols,
        evaluator, wandb_runs["tfidf"], scores, run_sim_plots,
    )

    # ── Word2Vec ──────────────────────────────────────────────────────────────
    models["w2v"], scores, metrics["w2v"] = _fit_w2v(
        cfg, fit_df, test_df, val_df, option_cols,
        evaluator, wandb_runs["word2vec"], scores,
    )

    # ── SBERT ────────────────────────────────────────────────────────────────
    models["sbert"], scores, metrics["sbert"] = _fit_sbert(
        cfg, val_df, test_df, option_cols,
        evaluator, wandb_runs["sbert"], scores, run_sim_plots,
    )

    return models, scores


# ── Per-model helpers (called by train_models) ─────────────────────
def _fit_tfidf(
    cfg, fit_df, val_df, test_df, option_cols,
    evaluator, wandb_run, scores, run_sim_plots,         # wandb_run = single run
):
    """Fit/load TF-IDF, score val+test, evaluate, log metrics."""
    ranker = TFIDFRanker(cfg)
    ranker.fit_or_load(fit_df)

    val_scores  = ranker.predict_scores(val_df)
    test_scores = ranker.predict_scores(test_df)
    scores["val"]["tfidf"]  = val_scores
    scores["test"]["tfidf"] = test_scores

    option_cols_present = [c for c in cfg.options if c in val_df.columns]
    val_preds = evaluator.scores_to_top_k_predictions(val_scores, option_cols_present)

    met = evaluator.evaluate(
        val_df, val_preds,
        answer_col=cfg.answer_col, split="tfidf_val", wandb_run=None,  # we log below
    )

    # ── Classification metrics + MAP → single W&B run ────────────────────────
    if cfg.answer_col in val_df.columns:
        actuals   = val_df[cfg.answer_col].tolist()
        clf_met   = evaluator.compute_classification_metrics(actuals, val_preds)
        map_score = evaluator.mean_average_precision_at_k(actuals, val_preds)

        log_model_metrics(wandb_run, {
            **clf_met,
            "map_at_k": map_score,
        })

    if run_sim_plots and cfg.answer_col in val_df.columns:
        corr, incorr = _split_sim_scores(
            val_scores, val_df[cfg.answer_col].tolist(), option_cols_present
        )
        plot_similarity_distributions(
            corr, incorr,
            method_name="TF-IDF", filename="tfidf_similarity_dist.png",
        )

    return ranker, scores, met


def _fit_w2v(
    cfg, fit_df, test_df, val_df, option_cols,
    evaluator, wandb_run, scores,
):
    """Train/load Word2Vec, score val+test, evaluate, log metrics."""
    ranker = Word2VecRanker(cfg)
    ranker.fit_or_load(fit_df, test_df)

    val_scores  = ranker.predict_scores(val_df)
    test_scores = ranker.predict_scores(test_df)
    scores["val"]["w2v"]  = val_scores
    scores["test"]["w2v"] = test_scores

    option_cols_present = [c for c in cfg.options if c in val_df.columns]
    val_preds = evaluator.scores_to_top_k_predictions(val_scores, option_cols_present)

    met = evaluator.evaluate(
        val_df, val_preds,
        answer_col=cfg.answer_col, split="w2v_val", wandb_run=None,
    )

    # ── Classification metrics + MAP → single W&B run ────────────────────────
    if cfg.answer_col in val_df.columns:
        actuals   = val_df[cfg.answer_col].tolist()
        clf_met   = evaluator.compute_classification_metrics(actuals, val_preds)
        map_score = evaluator.mean_average_precision_at_k(actuals, val_preds)

        log_model_metrics(wandb_run, {
            **clf_met,
            "map_at_k": map_score,
        })

    # PCA visualisation
    try:
        from sklearn.decomposition import PCA
        vocab_words = list(ranker.model.wv.key_to_index.keys())[:300]
        word_matrix = np.array([ranker.model.wv[w] for w in vocab_words])
        pca         = PCA(n_components=2, random_state=cfg.seed)
        reduced     = pca.fit_transform(word_matrix)
        plot_w2v_pca(reduced, vocab_words, n_label=50)
    except Exception as exc:
        logger.warning(f"W2V PCA plot skipped: {exc}")

    return ranker, scores, met


def _fit_sbert(
    cfg, val_df, test_df, option_cols,
    evaluator, wandb_run, scores, run_sim_plots,
):
    """Load SBERT, score val+test, evaluate, log metrics."""
    ranker = SBERTRanker(cfg)

    val_scores  = ranker.predict_scores(val_df)
    test_scores = ranker.predict_scores(test_df)
    scores["val"]["sbert"]  = val_scores
    scores["test"]["sbert"] = test_scores

    option_cols_present = [c for c in cfg.options if c in val_df.columns]
    val_preds = evaluator.scores_to_top_k_predictions(val_scores, option_cols_present)

    met = evaluator.evaluate(
        val_df, val_preds,
        answer_col=cfg.answer_col, split="sbert_val", wandb_run=None,
    )

    # ── Classification metrics + MAP → single W&B run ────────────────────────
    if cfg.answer_col in val_df.columns:
        actuals   = val_df[cfg.answer_col].tolist()
        clf_met   = evaluator.compute_classification_metrics(actuals, val_preds)
        map_score = evaluator.mean_average_precision_at_k(actuals, val_preds)

        log_model_metrics(wandb_run, {
            **clf_met,
            "map_at_k": map_score,
        })

    if run_sim_plots and cfg.answer_col in val_df.columns:
        corr, incorr = _split_sim_scores(
            val_scores, val_df[cfg.answer_col].tolist(), option_cols_present
        )
        plot_similarity_distributions(
            corr, incorr,
            method_name="SBERT", filename="sbert_similarity_dist.png",
        )

    return ranker, scores, met


def _fit_w2v(
    cfg, fit_df, test_df, val_df, option_cols,
    evaluator, wandb_run, scores,
):
    """Train/load Word2Vec, score val+test, evaluate, PCA plot."""
    ranker = Word2VecRanker(cfg)
    ranker.fit_or_load(fit_df, test_df)   # test_df adds vocab, no label leak

    val_scores  = ranker.predict_scores(val_df)
    test_scores = ranker.predict_scores(test_df)
    scores["val"]["w2v"]  = val_scores
    scores["test"]["w2v"] = test_scores

    val_preds = evaluator.scores_to_top_k_predictions(val_scores, option_cols)
    met = evaluator.evaluate(
        val_df, val_preds,
        answer_col=cfg.answer_col, split="w2v_val", wandb_run=wandb_run,
    )

    # PCA visualisation of word vectors
    try:
        from sklearn.decomposition import PCA
        vocab_words = list(ranker.model.wv.key_to_index.keys())[:300]
        word_matrix = np.array([ranker.model.wv[w] for w in vocab_words])
        pca         = PCA(n_components=2, random_state=cfg.seed)
        reduced     = pca.fit_transform(word_matrix)
        plot_w2v_pca(reduced, vocab_words, n_label=50)
    except Exception as exc:
        logger.warning(f"W2V PCA plot skipped: {exc}")

    return ranker, scores, met


def _fit_sbert(
    cfg, val_df, test_df, option_cols,
    evaluator, wandb_run, scores, run_sim_plots,
):
    """Load SBERT (pre-trained), score val+test, evaluate."""
    ranker = SBERTRanker(cfg)

    val_scores  = ranker.predict_scores(val_df)
    test_scores = ranker.predict_scores(test_df)
    scores["val"]["sbert"]  = val_scores
    scores["test"]["sbert"] = test_scores

    val_preds = evaluator.scores_to_top_k_predictions(val_scores, option_cols)
    met = evaluator.evaluate(
        val_df, val_preds,
        answer_col=cfg.answer_col, split="sbert_val", wandb_run=wandb_run,
    )

    if run_sim_plots and cfg.answer_col in val_df.columns:
        corr, incorr = _split_sim_scores(
            val_scores, val_df[cfg.answer_col].tolist(), option_cols
        )
        plot_similarity_distributions(
            corr, incorr,
            method_name="SBERT", filename="sbert_similarity_dist.png",
        )

    return ranker, scores, met


def _split_sim_scores(
    score_matrix : np.ndarray,
    actuals      : List[str],
    option_cols  : List[str],
) -> Tuple[List[float], List[float]]:
    """
    Split a score matrix into correct-option and incorrect-option lists.
    Used internally for similarity distribution plots.
    """
    correct, incorrect = [], []
    for i, actual in enumerate(actuals):
        for j, opt in enumerate(option_cols):
            (correct if opt == actual else incorrect).append(
                score_matrix[i, j]
            )
    return correct, incorrect


# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — Ensemble: grid-search weights → fuse → evaluate
# ══════════════════════════════════════════════════════════════════════
def run_ensemble(
    cfg        : Config,
    scores     : Dict[str, Dict[str, np.ndarray]],
    val_df     : "pd.DataFrame",
    option_cols: List[str],
    evaluator  : MAPAtKEvaluator,
    wandb_runs : Optional[Dict[str, Optional[object]]] = None,  # ← Dict now
    run_plots  : bool = True,
) -> Tuple[List[List[str]], List[List[str]], Dict[str, float]]:
    """Same logic — just pass wandb_runs dict; ensemble logs to tfidf run."""

    if wandb_runs is None:
        wandb_runs = {}

    actuals = val_df[cfg.answer_col].tolist()

    best_weights, best_map = ScoreFuser.grid_search(
        score_dict  = scores["val"],
        actuals     = actuals,
        evaluator   = evaluator,
        option_cols = option_cols,
    )

    fuser         = ScoreFuser(weights=best_weights)
    fused_val     = fuser.fuse(scores["val"])
    ens_val_preds = evaluator.scores_to_top_k_predictions(fused_val, option_cols)
    ens_metrics   = evaluator.evaluate(
        val_df, ens_val_preds,
        answer_col=cfg.answer_col, split="ensemble_val", wandb_run=None,
    )

    fused_test     = fuser.fuse(scores["test"])
    ens_test_preds = evaluator.scores_to_top_k_predictions(fused_test, option_cols)

    map_scores: Dict[str, float] = {
        "TF-IDF"  : evaluator.mean_average_precision_at_k(
                        actuals,
                        evaluator.scores_to_top_k_predictions(
                            scores["val"]["tfidf"], option_cols
                        )),
        "Word2Vec": evaluator.mean_average_precision_at_k(
                        actuals,
                        evaluator.scores_to_top_k_predictions(
                            scores["val"]["w2v"], option_cols
                        )),
        "SBERT"   : evaluator.mean_average_precision_at_k(
                        actuals,
                        evaluator.scores_to_top_k_predictions(
                            scores["val"]["sbert"], option_cols
                        )),
        "Ensemble": ens_metrics[f"ensemble_val/map@{cfg.top_k}"],
    }

    if run_plots:
        plot_map_comparison(map_scores, k=cfg.top_k)
        rank_dist: Dict = {1: 0, 2: 0, 3: 0, "not_found": 0}
        for actual, pred in zip(actuals, ens_val_preds):
            top = pred[: cfg.top_k]
            if actual in top:
                rank_dist[top.index(actual) + 1] += 1
            else:
                rank_dist["not_found"] += 1
        plot_rank_distribution(rank_dist, k=cfg.top_k)

    # ── Log summary to ALL three runs ────────────────────────────────────────
    if wandb_runs:
        import wandb
        summary_payload = {
            "phase1/tfidf_map3"   : map_scores["TF-IDF"],
            "phase1/w2v_map3"     : map_scores["Word2Vec"],
            "phase1/sbert_map3"   : map_scores["SBERT"],
            "phase1/ensemble_map3": map_scores["Ensemble"],
            "phase1/best_weights" : str(best_weights),
        }
        for run in wandb_runs.values():
            if run is not None:
                run.log(summary_payload)

    return ens_val_preds, ens_test_preds, map_scores

# ══════════════════════════════════════════════════════════════════════
# PHASE 4 — Submission file
# ══════════════════════════════════════════════════════════════════════

def generate_submission(
    cfg           : Config,
    test_df       : "pd.DataFrame",
    ens_test_preds: List[List[str]],
    filename      : str = "Milestone1_submission.csv",
) -> "pd.DataFrame":

    sub_gen    = SubmissionGenerator(cfg)
    submission = sub_gen.generate(test_df, ens_test_preds, filename=filename)
    return submission


# ══════════════════════════════════════════════════════════════════════
# PHASE 5 — Summary logging
# ══════════════════════════════════════════════════════════════════════
def print_summary(
    map_scores: Dict[str, float],
    cfg       : Config,
    wandb_runs: Optional[Dict[str, Optional[object]]] = None,  # ← Dict now
) -> None:
    """Log summary table and close all W&B runs."""
    sep = "═" * 52
    logger.info(f"\n{sep}\n  PHASE 1 SUMMARY\n{sep}")
    for name, score in map_scores.items():
        logger.info(f"  {name:<12}  MAP@{cfg.top_k} = {score:.4f}")
    logger.info(sep)

    # ── Close every run ───────────────────────────────────────────────────────
    if wandb_runs:
        for run in wandb_runs.values():
            finish_run(run)
        logger.info("All W&B runs closed.")


# ══════════════════════════════════════════════════════════════════════
# FULL PIPELINE — single call that chains all phases
# ══════════════════════════════════════════════════════════════════════
def run_full_pipeline(
    run_name           : str  = "phase1-baseline",
    run_eda_plots      : bool = True,
    run_sim_plots      : bool = True,
    run_ensemble_plots : bool = True,
    submission_filename: str  = "phase1_baseline_submission.csv",
) -> Dict[str, Any]:
    """Run the entire Phase-1 pipeline end-to-end."""

    # 0. Setup — now returns wandb_runs dict
    cfg, wandb_runs, evaluator = setup(run_name=run_name)

    # 1. Data — unchanged
    train_df, test_df, fit_df, val_df, option_cols = load_and_preprocess(
        cfg, run_eda_plots=run_eda_plots
    )

    # 2. Models — pass wandb_runs dict
    models, scores = train_models(
        cfg, fit_df, val_df, test_df,
        option_cols, evaluator, wandb_runs,          # ← dict
        run_sim_plots=run_sim_plots,
    )

    # 3. Ensemble — pass wandb_runs dict
    ens_val_preds, ens_test_preds, map_scores = run_ensemble(
        cfg, scores, val_df, option_cols, evaluator,
        wandb_runs,                                  # ← dict
        run_plots=run_ensemble_plots,
    )

    # 4. Submission — unchanged
    submission = generate_submission(
        cfg, test_df, ens_test_preds, filename=submission_filename
    )

    # 5. Summary — pass wandb_runs dict
    print_summary(map_scores, cfg, wandb_runs)       # ← dict

    return {
        "cfg"           : cfg,
        "wandb_runs"    : wandb_runs,                # ← dict
        "evaluator"     : evaluator,
        "train_df"      : train_df,
        "test_df"       : test_df,
        "fit_df"        : fit_df,
        "val_df"        : val_df,
        "option_cols"   : option_cols,
        "models"        : models,
        "scores"        : scores,
        "ens_val_preds" : ens_val_preds,
        "ens_test_preds": ens_test_preds,
        "map_scores"    : map_scores,
        "submission"    : submission,
    }

# ══════════════════════════════════════════════════════════════════════
# Script entry-point  (python main.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_full_pipeline()