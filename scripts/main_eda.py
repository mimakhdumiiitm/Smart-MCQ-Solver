# main.py

<<<<<<< HEAD
# ============================================================
# SETUP
# ============================================================
=======
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

>>>>>>> Milestone-1
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

<<<<<<< HEAD
# ============================================================
# CONFIG IMPORTS
# ============================================================
from config.config import (
    TEXT_COLS,
    OPTION_COLS,
    PROMPT_COL,
    ANSWER_COL,
    TOP_K,
    RANDOM_SEED,
    OUTPUT_DIR,
    TFIDF_INPUT_PATH,
    TFIDF_OUTPUT_PATH,
    MODEL_DIR,
    W2V_MODEL_PATH,
    PROCESSED_OUTPUT_DIR
)

# Create required directories
os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
from utils.data_loader import (
    load_train,
    load_test,
    validate_dataframe,
    check_missing_values,
    fill_missing_text,
    print_basic_stats,
)

# ============================================================
# PREPROCESSING
# ============================================================
from src.preprocessing import (
    clean_dataframe,
    add_text_length_features,
    build_row_corpus,
    build_token_sentences,
)

# ============================================================
# EMBEDDINGS
# ============================================================
from src.embeddings import (
    TFIDFEmbedder,
    Word2VecEmbedder,
    reduce_with_pca,
)

# ============================================================
# SIMILARITY
# ============================================================
from src.similarity import (
    tfidf_prompt_option_similarity,
    w2v_prompt_option_similarity,
    rank_options_by_similarity,
    inter_option_similarity_matrix,
    similarity_correct_vs_incorrect,
)

# ============================================================
# METRICS
# ============================================================
from utils.metrics import (
    evaluation_report,
    rank_distribution,
    compare_strategies,
)

# ============================================================
# VISUALIZATION
# ============================================================
from utils.visualization import (
=======
from config.config import Config, set_seed, get_logger

from utils.wandb_init import init_wandb
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessing import TextPreprocessor
from src.evaluation.evaluator import MAPAtKEvaluator
from src.models.tfidf_ranker import TFIDFRanker
from src.models.w2v_ranker import Word2VecRanker
from src.models.sbert_ranker import SBERTRanker
from src.ensemble.fuser import ScoreFuser
from utils.submission import SubmissionGenerator

from src.visualizaton.visualization import (
>>>>>>> Milestone-1
    plot_answer_distribution,
    plot_text_length_distributions,
    plot_top_words,
    plot_wordcloud,
    plot_similarity_distributions,
    plot_map_comparison,
    plot_rank_distribution,
    plot_w2v_pca,
)

<<<<<<< HEAD
from gensim.models import Word2Vec
# ============================================================
# MAIN EDA PIPELINE STARTS HERE
# ============================================================
print("All modules imported successfully.")

# ==================================================================
# STEP 1: LOAD DATA
# ==================================================================
def step_load_data():
    print("\n" + "=" * 50)
    print("STEP 1 : LOAD DATA")
    print("=" * 50)

    train_df = load_train()
    test_df  = load_test()

    validate_dataframe(train_df, TEXT_COLS + [ANSWER_COL], "Train")
    validate_dataframe(test_df,  TEXT_COLS,                "Test")

    print_basic_stats(train_df, "Train")
    print_basic_stats(test_df,  "Test")

    # Missing values
    missing = check_missing_values(train_df)
    if not missing.empty:
        print("\nMissing Values Report:")
        print(missing)
        train_df = fill_missing_text(train_df, TEXT_COLS)
        test_df  = fill_missing_text(test_df,  TEXT_COLS)
    else:
        print("No missing values in train set.")

    return train_df, test_df


# ==================================================================
# STEP 2: TEXT CLEANING & TOKENIZATION
# ==================================================================
def step_preprocessing(train_df: pd.DataFrame,
                        test_df:  pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 2 : TEXT CLEANING & TOKENIZATION")
    print("=" * 50)

    train_df = clean_dataframe(train_df, TEXT_COLS)
    test_df  = clean_dataframe(test_df,  TEXT_COLS)

    train_df = add_text_length_features(train_df, TEXT_COLS)
    test_df  = add_text_length_features(test_df,  TEXT_COLS)

    # Show before / after
    sample = train_df.iloc[0]
    print(f"\nOriginal prompt : {sample[PROMPT_COL][:100]}...")
    print(f"Cleaned  prompt : {sample[PROMPT_COL + '_clean'][:100]}...")

    # Text length plots
    plot_text_length_distributions(train_df, TEXT_COLS)

    return train_df, test_df


# ------------------------------------------------------------------
# Save processed DataFrames as CSVs
# ------------------------------------------------------------------
def save_processed_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save processed train/test DataFrames to OUTPUT_DIR as CSV files.

    Creates the directory if it does not exist.
    """
    out_dir = PROCESSED_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
=======
logger = get_logger("Main")


# ══════════════════════════════════════════════════════════════════════
# PHASE 0 — Config, seed, W&B
# ══════════════════════════════════════════════════════════════════════

def setup(
    run_name: str = "phase1-baseline",
    use_wandb_override: Optional[bool] = None,
) -> Tuple[Config, Optional[object], MAPAtKEvaluator]:
    """
    Initialise config, random seeds, W&B, and the shared evaluator.
>>>>>>> Milestone-1

    Parameters
    ----------
    run_name            : W&B run label.
    use_wandb_override  : Pass False to disable W&B without editing config.

    Returns
    -------
    cfg        : Config — pass this to every subsequent function.
    wandb_run  : Active W&B run or None.
    evaluator  : MAPAtKEvaluator — shared metric object.

<<<<<<< HEAD
    print(f"Saved processed DataFrames to: {out_dir}")


# ==================================================================
# STEP 3: WORD FREQUENCY & WORD CLOUDS
# ==================================================================
def step_word_frequency(train_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 3 : WORD FREQUENCY ANALYSIS")
    print("=" * 50)

    prompt_tokens = []
    option_tokens = []

    for _, row in train_df.iterrows():
        prompt_tokens.extend(str(row[PROMPT_COL + "_clean"]).split())
        for col in OPTION_COLS:
            option_tokens.extend(str(row[col + "_clean"]).split())

    prompt_freq = Counter(prompt_tokens)
    option_freq = Counter(option_tokens)

    print(f"Unique prompt tokens  : {len(prompt_freq)}")
    print(f"Unique option tokens  : {len(option_freq)}")
    print(f"Top 10 prompt words   : {prompt_freq.most_common(10)}")
    print(f"Top 10 option words   : {option_freq.most_common(10)}")

    plot_top_words(prompt_freq, title="Top 20 Prompt Words",
                   color="#2E86AB", filename="top_prompt_words.png")
    plot_top_words(option_freq, title="Top 20 Option Words",
                   color="#A23B72", filename="top_option_words.png")

    plot_wordcloud(prompt_tokens, title="Prompt Word Cloud",
                   colormap="Blues", filename="wordcloud_prompts.png")
    plot_wordcloud(option_tokens, title="Option Word Cloud",
                   colormap="Reds",  filename="wordcloud_options.png")

    return prompt_freq, option_freq


# ==================================================================
# STEP 4: ANSWER DISTRIBUTION
# ==================================================================
def step_answer_distribution(train_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 4 : ANSWER DISTRIBUTION")
    print("=" * 50)

    dist = train_df[ANSWER_COL].value_counts().sort_index()
    print(dist)
    plot_answer_distribution(train_df, answer_col=ANSWER_COL)


# ==================================================================
# STEP 5: TF-IDF EMBEDDINGS
# ==================================================================
#step 5 modified 
def step_tfidf(train_df, test_df):
=======
    Kaggle usage
    ------------
        cfg, wandb_run, evaluator = setup()
>>>>>>> Milestone-1
    """
    cfg = Config()
    set_seed(cfg.seed)

    if use_wandb_override is not None:
        cfg.use_wandb = use_wandb_override

    wandb_run = init_wandb(cfg, run_name=run_name)

    evaluator = MAPAtKEvaluator(k=cfg.top_k)
    MAPAtKEvaluator.run_unit_tests()

    logger.info("Setup complete.")
    return cfg, wandb_run, evaluator


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
    """
    Load raw CSVs → preprocess → EDA plots → stratified train/val split.

<<<<<<< HEAD
    # Reuse TF-IDF model from previous notebook if available
    if os.path.exists(TFIDF_INPUT_PATH):
        print(f"--> Reusing saved model from Kaggle Input: {TFIDF_INPUT_PATH}")
        embedder.load(TFIDF_INPUT_PATH)
    else:
        print("--> No saved model found. Fitting vocabulary from scratch...")
        train_corpus = build_row_corpus(train_df, cols=TEXT_COLS)
        embedder.fit(train_corpus)

    # Always write the TF-IDF model to the local model directory for each run
    os.makedirs(MODEL_DIR, exist_ok=True)
    embedder.save(TFIDF_OUTPUT_PATH)
=======
    Processed DataFrames are cached to CSV (controlled by
    ``cfg.use_cached_processed``).  Re-running this function a second
    time is therefore nearly instant.

    Parameters
    ----------
    cfg           : Config object from setup().
    run_eda_plots : Set False to skip matplotlib output (headless runs).

    Returns
    -------
    train_df, test_df, fit_df, val_df, option_cols

    Kaggle usage
    ------------
        train_df, test_df, fit_df, val_df, option_cols = load_and_preprocess(cfg)
    """
    # ── Raw data ────────────────────────────────────────────────
    loader       = DataLoader(cfg)
    raw_train_df = loader.load_train()
    raw_test_df  = loader.load_test()
    logger.info(f"Raw train: {raw_train_df.shape}  |  Raw test: {raw_test_df.shape}")
>>>>>>> Milestone-1

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

<<<<<<< HEAD
# ==================================================================
# STEP 6: WORD2VEC EMBEDDINGS
# ==================================================================
# step 6 - modified
def step_word2vec(train_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 6 : WORD2VEC EMBEDDINGS")
    print("=" * 50)

    # Use the clean, explicit path from your imported config module
    saved_model_path = W2V_MODEL_PATH

    if os.path.exists(saved_model_path):
        print(f"Found pre-saved model at: {saved_model_path}")
        w2v_embedder = Word2VecEmbedder()
        w2v_embedder.model = Word2Vec.load(saved_model_path)
        print(f"Word2Vec loaded successfully | vocab size : {len(w2v_embedder.model.wv)}")
    else:
        print("Pre-saved model not found! Falling back to training from scratch...")
        sentences = build_token_sentences(train_df, TEXT_COLS)
        print(f"Total training sentences : {len(sentences)}")

        w2v_embedder = Word2VecEmbedder()
        w2v_embedder.fit(sentences)

    # Always save the Word2Vec model locally on each run
    os.makedirs(MODEL_DIR, exist_ok=True)
    w2v_embedder.save(os.path.join(MODEL_DIR, "w2v.model"))

    # PCA visualization using config hyperparameters
    vocab_words = list(w2v_embedder.model.wv.key_to_index.keys())[:100]
    word_vecs   = np.array([w2v_embedder.get_word_vector(w) for w in vocab_words])
    reduced, _  = reduce_with_pca(word_vecs, n_components=2, seed=RANDOM_SEED) #config.
    plot_w2v_pca(reduced, vocab_words, n_label=40)

    return w2v_embedder


# ==================================================================
# STEP 7: COSINE SIMILARITY
# ==================================================================
def step_similarity(train_df:      pd.DataFrame,
                    tfidf_embedder: TFIDFEmbedder,
                    w2v_embedder:   Word2VecEmbedder):
    print("\n" + "=" * 50)
    print("STEP 7 : COSINE SIMILARITY")
    print("=" * 50)

    # TF-IDF similarity
    train_df = tfidf_prompt_option_similarity(
        train_df, tfidf_embedder,
        prompt_col=PROMPT_COL, option_cols=OPTION_COLS
    )

    # Word2Vec similarity
    train_df = w2v_prompt_option_similarity(
        train_df, w2v_embedder,
        prompt_col=PROMPT_COL, option_cols=OPTION_COLS
    )

    # Correct vs Incorrect analysis
    tfidf_result = similarity_correct_vs_incorrect(
        train_df, sim_prefix="tfidf_sim"
    )
    w2v_result = similarity_correct_vs_incorrect(
        train_df, sim_prefix="w2v_sim"
    )

    print(f"\nTF-IDF  | Correct mean : {tfidf_result['correct_mean']:.4f} "
          f"| Incorrect mean : {tfidf_result['incorrect_mean']:.4f}")
    print(f"Word2Vec | Correct mean : {w2v_result['correct_mean']:.4f} "
          f"| Incorrect mean : {w2v_result['incorrect_mean']:.4f}")

    # Plots
    plot_similarity_distributions(
        tfidf_result["correct"], tfidf_result["incorrect"],
        method_name="TF-IDF", filename="sim_dist_tfidf.png"
    )
    plot_similarity_distributions(
        w2v_result["correct"], w2v_result["incorrect"],
        method_name="Word2Vec", filename="sim_dist_w2v.png"
    )
    plot_mean_similarity_per_option(train_df, sim_prefix="tfidf_sim",
                                    title="Mean TF-IDF Similarity per Option")
    plot_mean_similarity_per_option(train_df, sim_prefix="w2v_sim",
                                    title="Mean Word2Vec Similarity per Option",
                                    filename="mean_sim_per_option_w2v.png")

    # Inter-option heatmap
    sim_matrix = inter_option_similarity_matrix(train_df, tfidf_embedder)
    plot_inter_option_heatmap(sim_matrix)

    return train_df


# ==================================================================
# STEP 8: MAP@3 EVALUATION
# ==================================================================
def step_metrics(train_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 8 : MAP@3 EVALUATION")
    print("=" * 50)

    actuals = train_df[ANSWER_COL].tolist()

    # --- Build prediction lists for each strategy ---

    # Random baseline
    np.random.seed(RANDOM_SEED)
    random_preds = [
        list(np.random.choice(OPTION_COLS, size=TOP_K, replace=False))
        for _ in range(len(train_df))
=======
    option_cols = [c for c in cfg.options if c in train_df.columns]

    return train_df, test_df, fit_df, val_df, option_cols


def _run_eda_plots(train_df: "pd.DataFrame", cfg: Config) -> None:
    """
    Internal helper — run all EDA visualisations.
    Called automatically by load_and_preprocess() when run_eda_plots=True.
    You can also call it directly if you want plots without re-loading data.

    Kaggle usage
    ------------
        _run_eda_plots(train_df, cfg)
    """
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
>>>>>>> Milestone-1
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
    cfg        : Config,
    fit_df     : "pd.DataFrame",
    val_df     : "pd.DataFrame",
    test_df    : "pd.DataFrame",
    option_cols: List[str],
    evaluator  : MAPAtKEvaluator,
    wandb_run  : Optional[object] = None,
    run_sim_plots: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    """
    Fit (or load from cache) TF-IDF, Word2Vec, and SBERT rankers.
    Evaluates each model on val_df and logs metrics to W&B.

    Parameters
    ----------
    cfg          : Config object.
    fit_df       : Training split (80 %).
    val_df       : Validation split (20 %).
    test_df      : Full test DataFrame.
    option_cols  : e.g. ["A","B","C","D","E"].
    evaluator    : shared MAPAtKEvaluator.
    wandb_run    : active W&B run or None.
    run_sim_plots: plot similarity distributions per model.

    Returns
    -------
    models : {"tfidf": TFIDFRanker, "w2v": Word2VecRanker, "sbert": SBERTRanker}
    scores : {
        "val" : {"tfidf": ndarray, "w2v": ndarray, "sbert": ndarray},
        "test": {"tfidf": ndarray, "w2v": ndarray, "sbert": ndarray},
    }

    Kaggle usage
    ------------
        models, scores = train_models(cfg, fit_df, val_df, test_df,
                                      option_cols, evaluator, wandb_run)
    """
    models : Dict[str, Any]                       = {}
    scores : Dict[str, Dict[str, np.ndarray]]     = {"val": {}, "test": {}}
    metrics: Dict[str, Dict[str, float]]          = {}

    # ── TF-IDF ──────────────────────────────────────────────────
    models["tfidf"], scores, metrics["tfidf"] = _fit_tfidf(
        cfg, fit_df, val_df, test_df, option_cols,
        evaluator, wandb_run, scores, run_sim_plots,
    )

    # ── Word2Vec ─────────────────────────────────────────────────
    models["w2v"], scores, metrics["w2v"] = _fit_w2v(
        cfg, fit_df, test_df, val_df, option_cols,
        evaluator, wandb_run, scores,
    )

    # ── SBERT ───────────────────────────────────────────────────
    models["sbert"], scores, metrics["sbert"] = _fit_sbert(
        cfg, val_df, test_df, option_cols,
        evaluator, wandb_run, scores, run_sim_plots,
    )

    return models, scores


# ── Per-model helpers (called by train_models) ─────────────────────

def _fit_tfidf(
    cfg, fit_df, val_df, test_df, option_cols,
    evaluator, wandb_run, scores, run_sim_plots,
):
    """Fit/load TF-IDF, score val+test, evaluate, optionally plot."""
    ranker = TFIDFRanker(cfg)
    ranker.fit_or_load(fit_df)

    val_scores  = ranker.predict_scores(val_df)
    test_scores = ranker.predict_scores(test_df)
    scores["val"]["tfidf"]  = val_scores
    scores["test"]["tfidf"] = test_scores

    val_preds = evaluator.scores_to_top_k_predictions(val_scores, option_cols)
    met = evaluator.evaluate(
        val_df, val_preds,
        answer_col=cfg.answer_col, split="tfidf_val", wandb_run=wandb_run,
    )

    if run_sim_plots and cfg.answer_col in val_df.columns:
        corr, incorr = _split_sim_scores(
            val_scores, val_df[cfg.answer_col].tolist(), option_cols
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
    wandb_run  : Optional[object] = None,
    run_plots  : bool = True,
) -> Tuple[List[List[str]], List[List[str]], Dict[str, float]]:
    """
    Grid-search best per-model weights → fuse val & test scores → evaluate.

    Parameters
    ----------
    cfg         : Config object.
    scores      : output of train_models()  {"val": {...}, "test": {...}}.
    val_df      : validation DataFrame (needs answer_col for ground truth).
    option_cols : option label list.
    evaluator   : shared MAPAtKEvaluator.
    wandb_run   : active W&B run or None.
    run_plots   : generate MAP comparison + rank distribution plots.

    Returns
    -------
    ens_val_preds  : ranked top-K labels for every val row.
    ens_test_preds : ranked top-K labels for every test row.
    map_scores     : {"TF-IDF": float, "Word2Vec": float,
                      "SBERT": float, "Ensemble": float}

    Kaggle usage
    ------------
        ens_val_preds, ens_test_preds, map_scores = run_ensemble(
            cfg, scores, val_df, option_cols, evaluator, wandb_run
        )
    """
    actuals = val_df[cfg.answer_col].tolist()

    # ── Grid-search best weights ─────────────────────────────────
    best_weights, best_map = ScoreFuser.grid_search(
        score_dict  = scores["val"],
        actuals     = actuals,
        evaluator   = evaluator,
        option_cols = option_cols,
    )

    # ── Fuse val scores ──────────────────────────────────────────
    fuser         = ScoreFuser(weights=best_weights)
    fused_val     = fuser.fuse(scores["val"])
    ens_val_preds = evaluator.scores_to_top_k_predictions(fused_val, option_cols)
    ens_metrics   = evaluator.evaluate(
        val_df, ens_val_preds,
        answer_col=cfg.answer_col, split="ensemble_val", wandb_run=wandb_run,
    )

    # ── Fuse test scores ─────────────────────────────────────────
    fused_test     = fuser.fuse(scores["test"])
    ens_test_preds = evaluator.scores_to_top_k_predictions(fused_test, option_cols)

    # ── Collect individual MAP scores for the summary chart ──────
    map_scores: Dict[str, float] = {
        "TF-IDF"  : evaluator.mean_average_precision_at_k(
                        actuals,
                        evaluator.scores_to_top_k_predictions(
                            scores["val"]["tfidf"], option_cols
                        )),
        "Word2Vec" : evaluator.mean_average_precision_at_k(
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

    # ── Visualisations ───────────────────────────────────────────
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

    # ── W&B summary ──────────────────────────────────────────────
    if wandb_run is not None:
        import wandb
        wandb_run.log({
            "phase1/tfidf_map3"    : map_scores["TF-IDF"],
            "phase1/w2v_map3"      : map_scores["Word2Vec"],
            "phase1/sbert_map3"    : map_scores["SBERT"],
            "phase1/ensemble_map3" : map_scores["Ensemble"],
            "phase1/best_weights"  : str(best_weights),
        })

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

    Parameters
    ----------
    cfg            : Config object.
    test_df        : processed test DataFrame (must have id column).
    ens_test_preds : top-K label lists from run_ensemble().
    filename       : output filename inside cfg.submission_dir.

    Returns
    -------
    pd.DataFrame  with columns [id, prediction].

    Kaggle usage
    ------------
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
    wandb_run : Optional[object] = None,
) -> None:
    """
    Log a formatted summary table and optionally close the W&B run.

    Parameters
    ----------
    map_scores : dict from run_ensemble().
    cfg        : Config object (used for top_k label).
    wandb_run  : if provided, wandb.finish() is called.

    Kaggle usage
    ------------
        print_summary(map_scores, cfg, wandb_run)
    """
    sep = "═" * 52
    logger.info(f"\n{sep}\n  PHASE 1 SUMMARY\n{sep}")
    for name, score in map_scores.items():
        logger.info(f"  {name:<12}  MAP@{cfg.top_k} = {score:.4f}")
    logger.info(sep)

    if wandb_run is not None:
        import wandb
        wandb_run.finish()
        logger.info("W&B run closed.")


# ══════════════════════════════════════════════════════════════════════
# FULL PIPELINE — single call that chains all phases
# ══════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    run_name          : str  = "phase1-baseline",
    run_eda_plots     : bool = True,
    run_sim_plots     : bool = True,
    run_ensemble_plots: bool = True,
    submission_filename: str = "phase1_baseline_submission.csv",
) -> Dict[str, Any]:
    """
    Run the entire Phase-1 pipeline end-to-end in a single call.

    Useful when you want zero-config execution; all knobs live in
    config/config.py.

    Parameters
    ----------
    run_name            : W&B run label.
    run_eda_plots       : generate EDA matplotlib plots.
    run_sim_plots       : generate per-model similarity distribution plots.
    run_ensemble_plots  : generate MAP comparison + rank distribution plots.
    submission_filename : output CSV name.

    Returns
    -------
    results dict with keys:
        cfg, wandb_run, evaluator,
        train_df, test_df, fit_df, val_df, option_cols,
        models, scores,
        ens_val_preds, ens_test_preds, map_scores,
        submission

    Kaggle usage
    ------------
        from main import run_full_pipeline
        results = run_full_pipeline()
    """
    # 0. Setup
    cfg, wandb_run, evaluator = setup(run_name=run_name)

    # 1. Data
    train_df, test_df, fit_df, val_df, option_cols = load_and_preprocess(
        cfg, run_eda_plots=run_eda_plots
    )

    # 2. Models
    models, scores = train_models(
        cfg, fit_df, val_df, test_df,
        option_cols, evaluator, wandb_run,
        run_sim_plots=run_sim_plots,
    )

    # 3. Ensemble
    ens_val_preds, ens_test_preds, map_scores = run_ensemble(
        cfg, scores, val_df, option_cols, evaluator, wandb_run,
        run_plots=run_ensemble_plots,
    )

    # 4. Submission
    submission = generate_submission(cfg, test_df, ens_test_preds,
                                     filename=submission_filename)

    # 5. Summary
    print_summary(map_scores, cfg, wandb_run)

    return {
        "cfg"            : cfg,
        "wandb_run"      : wandb_run,
        "evaluator"      : evaluator,
        "train_df"       : train_df,
        "test_df"        : test_df,
        "fit_df"         : fit_df,
        "val_df"         : val_df,
        "option_cols"    : option_cols,
        "models"         : models,
        "scores"         : scores,
        "ens_val_preds"  : ens_val_preds,
        "ens_test_preds" : ens_test_preds,
        "map_scores"     : map_scores,
        "submission"     : submission,
    }


# ══════════════════════════════════════════════════════════════════════
# Script entry-point  (python main.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_full_pipeline()