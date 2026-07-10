import os

# ============================================================
# SETUP
# ============================================================

import numpy as np
import pandas as pd
from collections import Counter

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
    plot_answer_distribution,
    plot_text_length_distributions,
    plot_top_words,
    plot_wordcloud,
    plot_similarity_distributions,
    plot_inter_option_heatmap,
    plot_mean_similarity_per_option,
    plot_map_comparison,
    plot_rank_distribution,
    plot_w2v_pca,
)

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

    train_path = os.path.join(out_dir, "train_processed.csv")
    test_path = os.path.join(out_dir, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

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
    """
    Handles TF-IDF embedding generation.
    Reuses a saved TF-IDF model if available; otherwise fits and saves a new one.
    """
    embedder = TFIDFEmbedder()

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

    # Transform train data
    print("Transforming training corpus...")
    train_corpus = build_row_corpus(train_df, cols=TEXT_COLS)
    X_train_tfidf = embedder.transform(train_corpus)

    # Transform test data
    print("Transforming testing corpus...")
    test_corpus = build_row_corpus(test_df, cols=TEXT_COLS)
    X_test_tfidf = embedder.transform(test_corpus)

    # Save processed DataFrames (creates OUTPUT_DIR if missing)
    try:
        save_processed_dataframes(train_df, test_df)
    except Exception as e:
        print(f"Warning: failed to save processed DataFrames: {e}")

    return X_train_tfidf, X_test_tfidf, embedder

# ==================================================================
# STEP 6: WORD2VEC EMBEDDINGS
# ==================================================================
# step 6 - modified
from gensim.models import Word2Vec

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
    ]

    # Always predict A B C
    always_abc_preds = [["A", "B", "C"]] * len(train_df)

    # TF-IDF cosine ranking
    tfidf_rank_col = rank_options_by_similarity(
        train_df, sim_prefix="tfidf_sim", top_k=TOP_K
    )
    tfidf_preds = [p.split() for p in tfidf_rank_col]

    # Word2Vec cosine ranking
    w2v_rank_col = rank_options_by_similarity(
        train_df, sim_prefix="w2v_sim", top_k=TOP_K
    )
    w2v_preds = [p.split() for p in w2v_rank_col]

    # Compare strategies
    strategies = {
        "Random"        : random_preds,
        "Always_A_B_C"  : always_abc_preds,
        "TF-IDF_cosine" : tfidf_preds,
        "Word2Vec_cosine": w2v_preds,
    }
    results_df = compare_strategies(strategies, actuals, k=TOP_K)
    print("\nStrategy Comparison:")
    print(results_df.to_string(index=False))

    # MAP@3 bar chart
    map_scores_dict = {
        row["strategy"]: row[f"map_at_{TOP_K}"]
        for _, row in results_df.iterrows()
    }
    plot_map_comparison(map_scores_dict, k=TOP_K)

    # Detailed report for TF-IDF
    print("\nDetailed Evaluation Report (TF-IDF):")
    report = evaluation_report(tfidf_preds, actuals, k=TOP_K)
    print(report.to_string(index=False))

    # Rank distribution
    dist = rank_distribution(tfidf_preds, actuals, k=TOP_K)
    print(f"\nRank Distribution (TF-IDF): {dist}")
    plot_rank_distribution(dist, k=TOP_K)

    return results_df
