# main.py

import logging
from collections import Counter
from pathlib import Path
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
# Score-artifact helpers
# ══════════════════════════════════════════════════════════════════════

def _try_load_scores(
    cfg      : Config,
    val_name : str,
    test_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cached score arrays using cfg.artifact_read_path().

    Priority (handled inside Config):
        1. Prebuilt Kaggle input artifact  (artifacts_load_dir / filename)
        2. Locally saved artifact          (artifacts_save_dir / filename)

    Returns (None, None) if neither artifact exists.
    """
    # Check existence first to avoid catching unrelated FileNotFoundErrors
    val_exists  = cfg.artifact_exists(val_name)
    test_exists = cfg.artifact_exists(test_name)

    if val_exists and test_exists:
        try:
            val_scores  = cfg.load_artifact(val_name)
            test_scores = cfg.load_artifact(test_name)
            logger.info(
                f"Reusing cached score arrays: {val_name}, {test_name}"
            )
            return val_scores, test_scores
        except Exception as exc:
            logger.warning(
                f"Score cache found but failed to load "
                f"({val_name}, {test_name}): {exc}"
            )
            return None, None

    logger.info(
        f"Score cache not found ({val_name}, {test_name}) — "
        f"will recompute."
    )
    return None, None


def _save_scores(
    cfg        : Config,
    val_scores : np.ndarray,
    test_scores: np.ndarray,
    val_name   : str,
    test_name  : str,
) -> None:
    """
    Save score arrays via cfg.save_artifact()
    (always writes to artifacts_save_dir).
    """
    cfg.save_artifact(val_scores,  val_name)
    cfg.save_artifact(test_scores, test_name)
    logger.info(
        f"Saved score artifacts to {cfg.artifacts_save_dir}: "
        f"{val_name}, {test_name}"
    )


# ══════════════════════════════════════════════════════════════════════
# PHASE 0 — Config, seed, W&B
# ══════════════════════════════════════════════════════════════════════

def setup(
    run_name          : str            = "phase1-baseline",
    use_wandb_override: Optional[bool] = None,
) -> Tuple[Config, MAPAtKEvaluator]:
    """
    Initialise config, random seeds, and evaluator.
    W&B runs are opened (and closed) inside each _fit_* helper,
    so a stale/finished run can never be passed to log().

    Kaggle cell usage
    -----------------
        cfg, evaluator = setup()
    """
    cfg = Config()
    set_seed(cfg.seed)

    if use_wandb_override is not None:
        cfg.use_wandb = use_wandb_override

    evaluator = MAPAtKEvaluator(k=cfg.top_k)
    MAPAtKEvaluator.run_unit_tests()

    logger.info("Setup complete.")
    return cfg, evaluator


# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — Data loading, preprocessing, EDA visuals, train/val split
# ══════════════════════════════════════════════════════════════════════

def load_and_preprocess(
    cfg          : Config,
    run_eda_plots: bool = True,
) -> Tuple[
    "pd.DataFrame",   # full processed train
    "pd.DataFrame",   # full processed test
    "pd.DataFrame",   # fit split  (80 %)
    "pd.DataFrame",   # val split  (20 %)
    List[str],        # option column labels  e.g. ["A","B","C","D","E"]
]:
    """
    Load raw CSVs → preprocess → EDA plots → stratified train/val split.

    Kaggle cell usage
    -----------------
        train_df, test_df, fit_df, val_df, option_cols = load_and_preprocess(cfg)
    """
    # ── Raw data ──────────────────────────────────────────────────────────────
    loader       = DataLoader(cfg)
    raw_train_df = loader.load_train()
    raw_test_df  = loader.load_test()
    logger.info(
        f"Raw train: {raw_train_df.shape}  |  Raw test: {raw_test_df.shape}"
    )

    # ── Preprocessing (cached) ────────────────────────────────────────────────
    prep     = TextPreprocessor()
    train_df = prep.process_dataframe(raw_train_df, cfg, split="train")
    test_df  = prep.process_dataframe(raw_test_df,  cfg, split="test")

    # ── EDA visualisations ────────────────────────────────────────────────────
    if run_eda_plots:
        _run_eda_plots(train_df, cfg)

    # ── Stratified train / val split ──────────────────────────────────────────
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
    """
    Run all EDA visualisations.

    Kaggle cell usage
    -----------------
        _run_eda_plots(train_df, cfg)
    """
    if cfg.answer_col in train_df.columns:
        plot_answer_distribution(train_df, answer_col=cfg.answer_col)

    plot_text_length_distributions(
        train_df, cols=[cfg.prompt_col] + cfg.options
    )

    all_tokens = [
        tok
        for text in train_df["prompt_clean"].fillna("")
        for tok in text.split()
    ]
    word_freq = Counter(all_tokens)
    plot_top_words(
        word_freq,
        title    = "Top Prompt Words",
        n        = 25,
        filename = "top_prompt_words.png",
    )
    plot_wordcloud(
        all_tokens,
        title    = "Prompt Word Cloud",
        filename = "prompt_wordcloud.png",
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
    run_sim_plots: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    """
    Fit/load TF-IDF, Word2Vec, and SBERT rankers; collect val+test scores.

    Score arrays are cached in artifacts_save_dir (or loaded from
    artifacts_load_dir if pre-built artifacts are available) via the
    Config.save_artifact / load_artifact helpers.

    Kaggle cell usage
    -----------------
        models, scores = train_models(
            cfg, fit_df, val_df, test_df, option_cols, evaluator
        )
    """
    models : Dict[str, Any]                   = {}
    scores : Dict[str, Dict[str, np.ndarray]] = {"val": {}, "test": {}}
    metrics: Dict[str, Dict[str, float]]      = {}

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    models["tfidf"], scores, metrics["tfidf"] = _fit_tfidf(
        cfg           = cfg,
        fit_df        = fit_df,
        val_df        = val_df,
        test_df       = test_df,
        option_cols   = option_cols,
        evaluator     = evaluator,
        scores        = scores,
        run_sim_plots = run_sim_plots,
    )

    # ── Word2Vec ──────────────────────────────────────────────────────────────
    models["w2v"], scores, metrics["w2v"] = _fit_w2v(
        cfg         = cfg,
        fit_df      = fit_df,
        test_df     = test_df,
        val_df      = val_df,
        option_cols = option_cols,
        evaluator   = evaluator,
        scores      = scores,
    )

    # ── SBERT ─────────────────────────────────────────────────────────────────
    models["sbert"], scores, metrics["sbert"] = _fit_sbert(
        cfg           = cfg,
        val_df        = val_df,
        test_df       = test_df,
        option_cols   = option_cols,
        evaluator     = evaluator,
        scores        = scores,
        run_sim_plots = run_sim_plots,
    )

    return models, scores


# ── Per-model private helpers ──────────────────────────────────────────────────

def _log_common_metrics(
    wandb_run : Optional[object],
    evaluator : MAPAtKEvaluator,
    actuals   : List[str],
    val_preds : List[List[str]],
    model_name: str,
) -> None:
    """
    Compute and log the shared metric set (accuracy, f1, precision, recall,
    map_at_k) to the supplied W&B run.
    """
    clf_met   = evaluator.compute_classification_metrics(actuals, val_preds)
    map_score = evaluator.mean_average_precision_at_k(actuals, val_preds)

    common_metrics = {
        "accuracy" : clf_met.get("accuracy",  0.0),
        "f1"       : clf_met.get("f1",        0.0),
        "precision": clf_met.get("precision", 0.0),
        "recall"   : clf_met.get("recall",    0.0),
        "map_at_k" : map_score,
        **clf_met,
    }

    log_model_metrics(wandb_run, common_metrics)
    logger.info(
        f"[{model_name}] accuracy={common_metrics['accuracy']:.4f}  "
        f"f1={common_metrics['f1']:.4f}  "
        f"map@k={map_score:.4f}"
    )


def _fit_tfidf(
    cfg          : Config,
    fit_df       : "pd.DataFrame",
    val_df       : "pd.DataFrame",
    test_df      : "pd.DataFrame",
    option_cols  : List[str],
    evaluator    : MAPAtKEvaluator,
    scores       : Dict[str, Dict[str, np.ndarray]],
    run_sim_plots: bool,
) -> Tuple[TFIDFRanker, Dict, Dict]:
    """
    Fit/load TF-IDF, score val+test, compute & log metrics.

    Score arrays are cached as
        artifacts_save_dir / tfidf_val_scores.npy
        artifacts_save_dir / tfidf_test_scores.npy
    and reused on subsequent runs (or loaded from artifacts_load_dir
    if pre-built Kaggle artifacts are present).

    Kaggle cell usage
    -----------------
        tfidf_ranker, scores, tfidf_met = _fit_tfidf(
            cfg, fit_df, val_df, test_df, option_cols,
            evaluator, scores, run_sim_plots=True
        )
    """
    wandb_run = init_wandb(cfg, run_name="phase1-tfidf", model_tag="tfidf")

    try:
        ranker = TFIDFRanker(cfg)

        # ── Try cached score arrays first ─────────────────────────────────────
        val_scores, test_scores = _try_load_scores(
            cfg, "tfidf_val_scores.npy", "tfidf_test_scores.npy"
        )

        if val_scores is None:
            # Scores not cached — fit (or load) model then encode
            ranker.fit_or_load(fit_df)
            val_scores  = ranker.predict_scores(val_df)
            test_scores = ranker.predict_scores(test_df)
            _save_scores(
                cfg, val_scores, test_scores,
                "tfidf_val_scores.npy", "tfidf_test_scores.npy",
            )
        else:
            # Scores already on disk — still need a fitted vectorizer for any
            # downstream callers that use ranker.vectorizer directly.
            ranker.fit_or_load(fit_df)

        scores["val"]["tfidf"]  = val_scores
        scores["test"]["tfidf"] = test_scores

        option_cols_present = [c for c in cfg.options if c in val_df.columns]
        val_preds = evaluator.scores_to_top_k_predictions(
            val_scores, option_cols_present
        )

        met = evaluator.evaluate(
            val_df, val_preds,
            answer_col=cfg.answer_col, split="tfidf_val", wandb_run=None,
        )

        if cfg.answer_col in val_df.columns:
            actuals = val_df[cfg.answer_col].tolist()
            _log_common_metrics(wandb_run, evaluator, actuals, val_preds, "TF-IDF")

            if run_sim_plots:
                corr, incorr = _split_sim_scores(
                    val_scores, actuals, option_cols_present
                )
                plot_similarity_distributions(
                    corr, incorr,
                    method_name = "TF-IDF",
                    filename    = "tfidf_similarity_dist.png",
                )

    finally:
        finish_run(wandb_run)
        logger.info("W&B run 'tfidf' closed.")

    return ranker, scores, met


def _fit_w2v(
    cfg        : Config,
    fit_df     : "pd.DataFrame",
    test_df    : "pd.DataFrame",
    val_df     : "pd.DataFrame",
    option_cols: List[str],
    evaluator  : MAPAtKEvaluator,
    scores     : Dict[str, Dict[str, np.ndarray]],
) -> Tuple[Word2VecRanker, Dict, Dict]:
    """
    Fit/load Word2Vec, score val+test, compute & log metrics.

    Score arrays are cached as
        artifacts_save_dir / w2v_val_scores.npy
        artifacts_save_dir / w2v_test_scores.npy
    and reused on subsequent runs (or loaded from artifacts_load_dir
    if pre-built Kaggle artifacts are present).

    Kaggle cell usage
    -----------------
        w2v_ranker, scores, w2v_met = _fit_w2v(
            cfg, fit_df, test_df, val_df, option_cols,
            evaluator, scores
        )
    """
    wandb_run = init_wandb(cfg, run_name="phase1-word2vec", model_tag="word2vec")

    try:
        ranker = Word2VecRanker(cfg)

        # ── Try cached score arrays first ─────────────────────────────────────
        val_scores, test_scores = _try_load_scores(
            cfg, "w2v_val_scores.npy", "w2v_test_scores.npy"
        )

        if val_scores is None:
            ranker.fit_or_load(fit_df, test_df)
            val_scores  = ranker.predict_scores(val_df)
            test_scores = ranker.predict_scores(test_df)
            _save_scores(
                cfg, val_scores, test_scores,
                "w2v_val_scores.npy", "w2v_test_scores.npy",
            )
        else:
            # Keep the model available for the PCA visualisation below
            ranker.fit_or_load(fit_df, test_df)

        scores["val"]["w2v"]  = val_scores
        scores["test"]["w2v"] = test_scores

        option_cols_present = [c for c in cfg.options if c in val_df.columns]
        val_preds = evaluator.scores_to_top_k_predictions(
            val_scores, option_cols_present
        )

        met = evaluator.evaluate(
            val_df, val_preds,
            answer_col=cfg.answer_col, split="w2v_val", wandb_run=None,
        )

        if cfg.answer_col in val_df.columns:
            actuals = val_df[cfg.answer_col].tolist()
            _log_common_metrics(wandb_run, evaluator, actuals, val_preds, "Word2Vec")

        # ── PCA visualisation ─────────────────────────────────────────────────
        try:
            from sklearn.decomposition import PCA
            vocab_words = list(ranker.model.wv.key_to_index.keys())[:300]
            word_matrix = np.array([ranker.model.wv[w] for w in vocab_words])
            pca         = PCA(n_components=2, random_state=cfg.seed)
            reduced     = pca.fit_transform(word_matrix)
            plot_w2v_pca(reduced, vocab_words, n_label=50)
        except Exception as exc:
            logger.warning(f"W2V PCA plot skipped: {exc}")

    finally:
        finish_run(wandb_run)
        logger.info("W&B run 'word2vec' closed.")

    return ranker, scores, met


def _fit_sbert(
    cfg          : Config,
    val_df       : "pd.DataFrame",
    test_df      : "pd.DataFrame",
    option_cols  : List[str],
    evaluator    : MAPAtKEvaluator,
    scores       : Dict[str, Dict[str, np.ndarray]],
    run_sim_plots: bool,
) -> Tuple[SBERTRanker, Dict, Dict]:
    """
    Load SBERT, score val+test, compute & log metrics.

    Score arrays are cached as
        artifacts_save_dir / sbert_val_scores.npy
        artifacts_save_dir / sbert_test_scores.npy
    and reused on subsequent runs (or loaded from artifacts_load_dir
    if pre-built Kaggle artifacts are present) — this avoids the
    expensive GPU encoding pass.

    Kaggle cell usage
    -----------------
        sbert_ranker, scores, sbert_met = _fit_sbert(
            cfg, val_df, test_df, option_cols,
            evaluator, scores, run_sim_plots=True
        )
    """
    wandb_run = init_wandb(cfg, run_name="phase1-sbert", model_tag="sbert")

    try:
        ranker = SBERTRanker(cfg)

        # ── Try cached score arrays first ─────────────────────────────────────
        val_scores, test_scores = _try_load_scores(
            cfg, "sbert_val_scores.npy", "sbert_test_scores.npy"
        )

        if val_scores is None:
            # No cache — run the (expensive) encoding pass and save
            val_scores  = ranker.predict_scores(val_df)
            test_scores = ranker.predict_scores(test_df)
            _save_scores(
                cfg, val_scores, test_scores,
                "sbert_val_scores.npy", "sbert_test_scores.npy",
            )
        # Note: no else branch needed — SBERT has no extra state to restore
        # because predict_scores() uses the pre-trained model directly.

        scores["val"]["sbert"]  = val_scores
        scores["test"]["sbert"] = test_scores

        option_cols_present = [c for c in cfg.options if c in val_df.columns]
        val_preds = evaluator.scores_to_top_k_predictions(
            val_scores, option_cols_present
        )

        met = evaluator.evaluate(
            val_df, val_preds,
            answer_col=cfg.answer_col, split="sbert_val", wandb_run=None,
        )

        if cfg.answer_col in val_df.columns:
            actuals = val_df[cfg.answer_col].tolist()
            _log_common_metrics(wandb_run, evaluator, actuals, val_preds, "SBERT")

            if run_sim_plots:
                corr, incorr = _split_sim_scores(
                    val_scores, actuals, option_cols_present
                )
                plot_similarity_distributions(
                    corr, incorr,
                    method_name = "SBERT",
                    filename    = "sbert_similarity_dist.png",
                )

    finally:
        finish_run(wandb_run)
        logger.info("W&B run 'sbert' closed.")

    return ranker, scores, met


def _split_sim_scores(
    score_matrix: np.ndarray,
    actuals     : List[str],
    option_cols : List[str],
) -> Tuple[List[float], List[float]]:
    """Split score matrix into correct / incorrect option score lists."""
    correct, incorrect = [], []
    for i, actual in enumerate(actuals):
        for j, opt in enumerate(option_cols):
            (correct if opt == actual else incorrect).append(score_matrix[i, j])
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
    run_plots  : bool = True,
) -> Tuple[List[List[str]], List[List[str]], Dict[str, float]]:
    """
    Grid-search best weights → fuse val & test scores → evaluate.

    Returns
    -------
    ens_val_preds  : ranked top-K labels for every val row
    ens_test_preds : ranked top-K labels for every test row
    map_scores     : {"TF-IDF": float, "Word2Vec": float,
                      "SBERT": float, "Ensemble": float}

    Kaggle cell usage
    -----------------
        ens_val_preds, ens_test_preds, map_scores = run_ensemble(
            cfg, scores, val_df, option_cols, evaluator
        )
    """
    actuals = val_df[cfg.answer_col].tolist()

    # ── Grid-search best weights ──────────────────────────────────────────────
    best_weights, best_map = ScoreFuser.grid_search(
        score_dict  = scores["val"],
        actuals     = actuals,
        evaluator   = evaluator,
        option_cols = option_cols,
    )

    # ── Fuse val scores ───────────────────────────────────────────────────────
    fuser         = ScoreFuser(weights=best_weights)
    fused_val     = fuser.fuse(scores["val"])
    ens_val_preds = evaluator.scores_to_top_k_predictions(fused_val, option_cols)
    ens_metrics   = evaluator.evaluate(
        val_df, ens_val_preds,
        answer_col=cfg.answer_col, split="ensemble_val", wandb_run=None,
    )

    # ── Fuse test scores ──────────────────────────────────────────────────────
    fused_test     = fuser.fuse(scores["test"])
    ens_test_preds = evaluator.scores_to_top_k_predictions(fused_test, option_cols)

    # ── Collect individual MAP@K for summary chart ────────────────────────────
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

    # ── Visualisations ────────────────────────────────────────────────────────
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
    """
    Write the Kaggle submission CSV and return it as a DataFrame.

    Kaggle cell usage
    -----------------
        submission = generate_submission(cfg, test_df, ens_test_preds)
    """
    sub_gen    = SubmissionGenerator(cfg)
    submission = sub_gen.generate(test_df, ens_test_preds, filename=filename)
    return submission


# ══════════════════════════════════════════════════════════════════════
# PHASE 5 — Summary logging
# ══════════════════════════════════════════════════════════════════════

def print_summary(
    map_scores: Dict[str, float],
    cfg       : Config,
) -> None:
    """
    Print final MAP@K table.
    W&B runs are already closed inside each _fit_* helper.
    """
    sep = "═" * 52
    logger.info(f"\n{sep}\n  PHASE 1 SUMMARY\n{sep}")
    for name, score in map_scores.items():
        logger.info(f"  {name:<12}  MAP@{cfg.top_k} = {score:.4f}")
    logger.info(sep)


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
    """
    Chain all phases into a single end-to-end pipeline call.

    Kaggle cell usage
    -----------------
        results = run_full_pipeline()
    """
    # 0. Setup
    cfg, evaluator = setup(run_name=run_name)

    # 1. Data
    train_df, test_df, fit_df, val_df, option_cols = load_and_preprocess(
        cfg, run_eda_plots=run_eda_plots
    )

    # 2. Models — each _fit_* opens/closes its own W&B run;
    #             score arrays are cached under cfg.artifacts_save_dir
    #             (or loaded from cfg.artifacts_load_dir if pre-built)
    models, scores = train_models(
        cfg           = cfg,
        fit_df        = fit_df,
        val_df        = val_df,
        test_df       = test_df,
        option_cols   = option_cols,
        evaluator     = evaluator,
        run_sim_plots = run_sim_plots,
    )

    # 3. Ensemble
    ens_val_preds, ens_test_preds, map_scores = run_ensemble(
        cfg         = cfg,
        scores      = scores,
        val_df      = val_df,
        option_cols = option_cols,
        evaluator   = evaluator,
        run_plots   = run_ensemble_plots,
    )

    # 4. Submission
    submission = generate_submission(
        cfg, test_df, ens_test_preds, filename=submission_filename
    )

    # 5. Summary
    print_summary(map_scores, cfg)

    return {
        "cfg"           : cfg,
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