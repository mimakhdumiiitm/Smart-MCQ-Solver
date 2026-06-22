# similarity.py
# Cosine similarity computation between prompts and answer options

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config.config       import OPTION_COLS, PROMPT_COL
from preprocessing import clean_text


# ------------------------------------------------------------------
# CORE SIMILARITY FUNCTION
# ------------------------------------------------------------------

def cosine_sim(vec_a: np.ndarray,
               vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D or 2-D numpy arrays.

    Formula:
        cos(A, B) = (A . B) / (||A|| * ||B||)

    Returns a float in [-1, 1].

    Usage:
        score = cosine_sim(vec1, vec2)
    """
    vec_a = np.asarray(vec_a).reshape(1, -1)
    vec_b = np.asarray(vec_b).reshape(1, -1)
    return float(cosine_similarity(vec_a, vec_b)[0][0])


# ------------------------------------------------------------------
# TF-IDF SIMILARITY
# ------------------------------------------------------------------

def tfidf_prompt_option_similarity(df: pd.DataFrame,
                                   tfidf_embedder,
                                   prompt_col:  str  = PROMPT_COL,
                                   option_cols: list = None,
                                   col_suffix:  str  = "tfidf_sim") -> pd.DataFrame:
    """
    For each row, compute the cosine similarity between the prompt
    TF-IDF vector and each answer option's TF-IDF vector.

    Adds new columns to the DataFrame:
        tfidf_sim_A, tfidf_sim_B, tfidf_sim_C, tfidf_sim_D, tfidf_sim_E

    Args:
        df             : input DataFrame
        tfidf_embedder : fitted TFIDFEmbedder instance
        prompt_col     : name of the prompt column
        option_cols    : list of option column names
        col_suffix     : prefix for new similarity columns

    Returns:
        DataFrame with similarity columns added.

    Usage:
        train_df = tfidf_prompt_option_similarity(train_df, tfidf_embedder)
        # Access via: train_df["tfidf_sim_A"]
    """
    if option_cols is None:
        option_cols = OPTION_COLS

    df = df.copy()

    for col in option_cols:
        if col not in df.columns:
            continue

        sims = []
        for _, row in df.iterrows():
            prompt_vec = tfidf_embedder.transform_one(str(row[prompt_col]))
            option_vec = tfidf_embedder.transform_one(str(row[col]))
            sims.append(
                float(cosine_similarity(prompt_vec, option_vec)[0][0])
            )

        new_col       = f"{col_suffix}_{col}"
        df[new_col]   = sims
        print(f"Computed {new_col}")

    return df


# ------------------------------------------------------------------
# WORD2VEC SIMILARITY
# ------------------------------------------------------------------

def w2v_prompt_option_similarity(df: pd.DataFrame,
                                 w2v_embedder,
                                 prompt_col:  str  = PROMPT_COL,
                                 option_cols: list = None,
                                 col_suffix:  str  = "w2v_sim") -> pd.DataFrame:
    """
    For each row, compute the cosine similarity between the prompt
    Word2Vec document vector and each answer option's document vector.

    Adds new columns to the DataFrame:
        w2v_sim_A, w2v_sim_B, w2v_sim_C, w2v_sim_D, w2v_sim_E

    Args:
        df           : input DataFrame
        w2v_embedder : fitted Word2VecEmbedder instance
        prompt_col   : name of the prompt column
        option_cols  : list of option column names
        col_suffix   : prefix for new similarity columns

    Returns:
        DataFrame with similarity columns added.

    Usage:
        train_df = w2v_prompt_option_similarity(train_df, w2v_embedder)
        # Access via: train_df["w2v_sim_A"]
    """
    if option_cols is None:
        option_cols = OPTION_COLS

    df = df.copy()

    for col in option_cols:
        if col not in df.columns:
            continue

        sims = []
        for _, row in df.iterrows():
            prompt_vec = w2v_embedder.get_doc_vector(str(row[prompt_col]))
            option_vec = w2v_embedder.get_doc_vector(str(row[col]))
            sims.append(cosine_sim(prompt_vec, option_vec))

        new_col     = f"{col_suffix}_{col}"
        df[new_col] = sims
        print(f"Computed {new_col}")

    return df


# ------------------------------------------------------------------
# SIMILARITY-BASED RANKING
# ------------------------------------------------------------------

def rank_options_by_similarity(df: pd.DataFrame,
                               sim_prefix: str  = "tfidf_sim",
                               option_cols: list = None,
                               top_k: int        = 3) -> pd.Series:
    """
    For each row, rank the answer options by their similarity score
    and return the top-k as a space-separated string.

    This can be used directly as a baseline prediction.

    Args:
        df          : DataFrame containing similarity columns
        sim_prefix  : prefix of the similarity columns (e.g. "tfidf_sim")
        option_cols : list of option labels
        top_k       : number of top options to return

    Returns:
        pd.Series of strings, e.g. "B A C" for each row.

    Usage:
        train_df["tfidf_pred"] = rank_options_by_similarity(
            train_df, sim_prefix="tfidf_sim", top_k=3
        )
    """
    if option_cols is None:
        option_cols = OPTION_COLS

    sim_cols = [f"{sim_prefix}_{col}" for col in option_cols]

    # Verify columns exist
    missing = [c for c in sim_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing similarity columns: {missing}\n"
            f"Run the similarity computation first."
        )

    def rank_row(row):
        scores = {col: row[f"{sim_prefix}_{col}"] for col in option_cols}
        ranked = sorted(scores, key=scores.get, reverse=True)
        return " ".join(ranked[:top_k])

    return df.apply(rank_row, axis=1)


# ------------------------------------------------------------------
# INTER-OPTION SIMILARITY
# ------------------------------------------------------------------

def inter_option_similarity_matrix(df: pd.DataFrame,
                                   tfidf_embedder,
                                   option_cols: list = None) -> np.ndarray:
    """
    Compute the average pairwise cosine similarity between all
    answer options across all rows.

    Returns an (n_options x n_options) matrix.

    Usage:
        matrix = inter_option_similarity_matrix(train_df, tfidf_embedder)
        # matrix[0][1] = mean cosine similarity between option A and B
    """
    if option_cols is None:
        option_cols = OPTION_COLS

    n  = len(option_cols)
    sim_matrix = np.zeros((n, n))

    for i, col_i in enumerate(option_cols):
        for j, col_j in enumerate(option_cols):
            sims = []
            for _, row in df.iterrows():
                v_i = tfidf_embedder.transform_one(str(row[col_i]))
                v_j = tfidf_embedder.transform_one(str(row[col_j]))
                sims.append(
                    float(cosine_similarity(v_i, v_j)[0][0])
                )
            sim_matrix[i, j] = np.mean(sims)

    return sim_matrix


# ------------------------------------------------------------------
# CORRECT vs INCORRECT SIMILARITY ANALYSIS
# ------------------------------------------------------------------

def similarity_correct_vs_incorrect(df: pd.DataFrame,
                                    sim_prefix:  str  = "tfidf_sim",
                                    option_cols: list = None,
                                    answer_col:  str  = "answer") -> dict:
    """
    Separate similarity scores into correct-answer and incorrect-answer
    groups and return summary statistics.

    Args:
        df          : DataFrame with similarity columns and answer column
        sim_prefix  : prefix of similarity columns
        option_cols : list of option labels
        answer_col  : name of the answer column

    Returns:
        dict with keys 'correct' and 'incorrect', each containing
        a list of similarity scores.

    Usage:
        result = similarity_correct_vs_incorrect(train_df, "tfidf_sim")
        print("Correct mean  :", np.mean(result["correct"]))
        print("Incorrect mean:", np.mean(result["incorrect"]))
    """
    if option_cols is None:
        option_cols = OPTION_COLS

    if answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' not found in DataFrame.")

    correct_sims   = []
    incorrect_sims = []

    for _, row in df.iterrows():
        for col in option_cols:
            sim_col = f"{sim_prefix}_{col}"
            if sim_col in row:
                if row[answer_col] == col:
                    correct_sims.append(row[sim_col])
                else:
                    incorrect_sims.append(row[sim_col])

    return {
        "correct"  : correct_sims,
        "incorrect": incorrect_sims,
        "correct_mean"  : float(np.mean(correct_sims))   if correct_sims   else 0.0,
        "incorrect_mean": float(np.mean(incorrect_sims)) if incorrect_sims else 0.0,
    }