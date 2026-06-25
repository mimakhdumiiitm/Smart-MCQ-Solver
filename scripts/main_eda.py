
# main_eda.py
# ============================================================
# SETUP
# ============================================================

import os
import sys

# ============================================================
# STANDARD LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
from collections import Counter

# ============================================================
# CONFIG IMPORTS
# ============================================================

from config.config import (
    TRAIN_PATH,
    TEST_PATH,
    TEXT_COLS,
    OPTION_COLS,
    PROMPT_COL,
    ANSWER_COL,
    TOP_K,
    RANDOM_SEED,
    OUTPUT_DIR
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
    print_basic_stats
)

# ============================================================
# PREPROCESSING
# ============================================================

from src.preprocessing import (
    clean_text,
    clean_dataframe,
    add_text_length_features,
    build_row_corpus,
    build_token_sentences
)

# ============================================================
# EMBEDDINGS
# ============================================================

from src.embeddings import (
    TFIDFEmbedder,
    Word2VecEmbedder,
    top_features_from_matrix,
    reduce_with_pca,
    reduce_with_svd
)

# ============================================================
# SIMILARITY
# ============================================================

from src.similarity import (
    tfidf_prompt_option_similarity,
    w2v_prompt_option_similarity,
    rank_options_by_similarity,
    inter_option_similarity_matrix,
    similarity_correct_vs_incorrect
)

# ============================================================
# METRICS
# ============================================================

from utils.metrics import (
    map_at_k,
    evaluation_report,
    rank_distribution,
    compare_strategies,
    format_submission,
    parse_predictions_series
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
    plot_w2v_pca
)

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

import os
import joblib  # or 'import pickle' depending on how your TFIDFEmbedder.save works
def step_tfidf(train_df: pd.DataFrame, 
               test_df:  pd.DataFrame, 
               pretrained_path: str = None):
    print("\n" + "=" * 50)
    print("STEP 5 : TF-IDF EMBEDDINGS")
    print("=" * 50)

    # 1. Check if a pre-trained model path is provided and exists
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained TF-IDF model from: {pretrained_path}")
        # If your class has a built-in load method, use it:
        tfidf_embedder = TFIDFEmbedder.load(pretrained_path) 
        # Alternatively, if it uses standard joblib/pickle:
        # tfidf_embedder = joblib.load(pretrained_path)
        
        return tfidf_embedder

    # 2. If no pre-trained model is found, fall back to training it
    print("No pre-trained model found. Fitting a new TF-IDF model...")
    corpus = build_row_corpus(train_df, TEXT_COLS, clean=True)

    tfidf_embedder = TFIDFEmbedder()
    tfidf_matrix   = tfidf_embedder.fit_transform(corpus)

    print(f"Matrix shape : {tfidf_matrix.shape}")
    print(f"Sparsity     : "
          f"{100*(1 - tfidf_matrix.nnz/np.prod(tfidf_matrix.shape)):.2f}%")

    top_df = top_features_from_matrix(
        tfidf_matrix, tfidf_embedder.get_feature_names(), n=20
    )
    print("\nTop 10 TF-IDF Features:")
    print(top_df.head(10).to_string(index=False))

    # Keep this here so future runs can save it to the current working directory
    os.makedirs("models", exist_ok=True)
    tfidf_embedder.save(os.path.join("models", "tfidf.pkl"))

    return tfidf_embedder


# ==================================================================
# STEP 6: WORD2VEC EMBEDDINGS
# ==================================================================

def step_word2vec(train_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 6 : WORD2VEC EMBEDDINGS")
    print("=" * 50)

    sentences = build_token_sentences(train_df, TEXT_COLS)
    print(f"Total training sentences : {len(sentences)}")

    w2v_embedder = Word2VecEmbedder()
    w2v_embedder.fit(sentences)

    # Similarity examples
    test_words = ["time", "fusion", "energy", "theory"]
    for word in test_words:
        similar = w2v_embedder.most_similar(word, topn=3)
        if similar:
            print(f"  '{word}' similar to : {similar}")

    # PCA visualization
    vocab_words = list(w2v_embedder.model.wv.key_to_index.keys())[:100]
    word_vecs   = np.array([w2v_embedder.get_word_vector(w) for w in vocab_words])
    reduced, _  = reduce_with_pca(word_vecs, n_components=2, seed=RANDOM_SEED)
    plot_w2v_pca(reduced, vocab_words, n_label=40)

    # Save model
    w2v_embedder.save(os.path.join("models", "w2v.model"))

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
        prompt_col="prompt", option_cols=OPTION_COLS
    )

    # Word2Vec similarity
    train_df = w2v_prompt_option_similarity(
        train_df, w2v_embedder,
        prompt_col="prompt", option_cols=OPTION_COLS
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
