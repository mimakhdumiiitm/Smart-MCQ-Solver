# preprocessing.py
# All text cleaning, tokenization, and feature engineering functions

import re
import string
import pandas as pd
import numpy as np

import nltk
nltk.download("punkt",                    quiet=True)
nltk.download("punkt_tab",               quiet=True)
nltk.download("stopwords",               quiet=True)
nltk.download("wordnet",                 quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer, PorterStemmer

from config.config import (
    OPTION_COLS, TEXT_COLS, PROMPT_COL,
    REMOVE_STOPWORDS, LEMMATIZE, STEM, MIN_TOKEN_LENGTH
)


# ------------------------------------------------------------------
# SHARED OBJECTS  (instantiate once, reuse everywhere)
# ------------------------------------------------------------------
STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()
STEMMER     = PorterStemmer()


# ------------------------------------------------------------------
# LOW-LEVEL CLEANING HELPERS
# ------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www addresses."""
    return re.sub(r"http\S+|www\S+|https\S+", " ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r"\S+@\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """Keep only letters, digits, and whitespace."""
    return re.sub(r"[^a-z0-9\s]", " ", text)


def remove_digits(text: str) -> str:
    """Remove standalone digit sequences."""
    return re.sub(r"\b\d+\b", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Tokenize text using NLTK word_tokenize.
    
    Usage:
        tokens = tokenize("Martin Heidegger's view on time.")
        # ['Martin', 'Heidegger', "'s", 'view', 'on', 'time', '.']
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    return word_tokenize(text)


def tokenize_clean(text: str) -> list:
    """
    Tokenize already-cleaned (lowercase) text, filtering short tokens.
    
    Usage:
        tokens = tokenize_clean("martin heidegger view time")
        # ['martin', 'heidegger', 'view', 'time']
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    return [
        t for t in word_tokenize(text)
        if len(t) >= MIN_TOKEN_LENGTH
    ]


# ------------------------------------------------------------------
# FULL CLEANING PIPELINE
# ------------------------------------------------------------------

def clean_text(text: str,
               remove_stopwords: bool = REMOVE_STOPWORDS,
               lemmatize: bool        = LEMMATIZE,
               stem: bool             = STEM) -> str:
    """
    Full NLP text cleaning pipeline:
        1. Lowercase
        2. Remove URLs, emails
        3. Remove special characters
        4. Remove digits
        5. Tokenize
        6. Remove stopwords (optional)
        7. Lemmatize (optional)
        8. Stem (optional, applied after lemmatize)
        9. Rejoin tokens

    Args:
        text             : raw input string
        remove_stopwords : whether to drop NLTK English stopwords
        lemmatize        : whether to apply WordNet lemmatization
        stem             : whether to apply Porter stemming

    Returns:
        Cleaned string.

    Usage:
        cleaned = clean_text("What is Martin Heidegger's view on time?")
        # "martin heidegger view time"
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and emails
    text = remove_urls(text)
    text = remove_emails(text)

    # 3. Remove special characters
    text = remove_special_characters(text)

    # 4. Remove digit-only tokens
    text = remove_digits(text)

    # 5. Normalize whitespace
    text = normalize_whitespace(text)

    # 6. Tokenize
    tokens = word_tokenize(text)

    # 7. Filter short tokens
    tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]

    # 8. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # 9. Lemmatize
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    # 10. Stem
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


# ------------------------------------------------------------------
# BATCH CLEANING
# ------------------------------------------------------------------

def clean_series(series: pd.Series,
                 remove_stopwords: bool = REMOVE_STOPWORDS,
                 lemmatize: bool        = LEMMATIZE,
                 stem: bool             = STEM) -> pd.Series:
    """
    Apply clean_text to every element of a pandas Series.
    
    Usage:
        train_df["prompt_clean"] = clean_series(train_df["prompt"])
    """
    return series.apply(
        lambda x: clean_text(x, remove_stopwords, lemmatize, stem)
    )


def clean_dataframe(df: pd.DataFrame,
                    cols: list         = None,
                    suffix: str        = "_clean",
                    remove_stopwords: bool = REMOVE_STOPWORDS,
                    lemmatize: bool        = LEMMATIZE,
                    stem: bool             = STEM) -> pd.DataFrame:
    """
    Apply clean_text to specified columns and store results as
    new columns with a suffix appended.

    Args:
        df               : input DataFrame
        cols             : list of column names to clean (default: TEXT_COLS)
        suffix           : suffix for new cleaned columns
        remove_stopwords : passed to clean_text
        lemmatize        : passed to clean_text
        stem             : passed to clean_text

    Returns:
        DataFrame with additional cleaned columns.

    Usage:
        train_df = clean_dataframe(train_df, cols=TEXT_COLS)
        # Creates: prompt_clean, A_clean, B_clean, ...
    """
    if cols is None:
        cols = TEXT_COLS

    df = df.copy()
    for col in cols:
        if col in df.columns:
            new_col     = f"{col}{suffix}"
            df[new_col] = clean_series(
                df[col], remove_stopwords, lemmatize, stem
            )
            print(f"Cleaned column '{col}' -> '{new_col}'")
    return df


# ------------------------------------------------------------------
# TEXT STATISTICS
# ------------------------------------------------------------------

def text_length_stats(text: str) -> dict:
    """
    Compute character count, raw word count, and unique word count
    for a single string.
    
    Usage:
        stats = text_length_stats("What is fusion?")
        # {'char_count': 15, 'word_count': 3, 'unique_words': 3}
    """
    if not isinstance(text, str):
        return {"char_count": 0, "word_count": 0, "unique_words": 0}

    tokens = word_tokenize(text.lower())
    return {
        "char_count"  : len(text),
        "word_count"  : len(tokens),
        "unique_words": len(set(tokens))
    }


def add_text_length_features(df: pd.DataFrame,
                              cols: list = None) -> pd.DataFrame:
    """
    Add character count and word count columns for each specified column.

    Usage:
        train_df = add_text_length_features(train_df, TEXT_COLS)
        # Creates: prompt_char_len, prompt_word_len, A_char_len, ...
    """
    if cols is None:
        cols = TEXT_COLS

    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_char_len"] = df[col].apply(
                lambda x: len(str(x))
            )
            df[f"{col}_word_len"] = df[col].apply(
                lambda x: len(str(x).split())
            )
    return df


# ------------------------------------------------------------------
# CORPUS BUILDER (for embedding training)
# ------------------------------------------------------------------

def build_corpus(df: pd.DataFrame,
                 cols: list = None,
                 clean: bool = True) -> list:
    """
    Build a flat list of cleaned text strings from DataFrame columns.
    One string per (row, column) combination.
    Useful as input for TF-IDF or Word2Vec training.

    Args:
        df    : input DataFrame
        cols  : columns to include (default: TEXT_COLS)
        clean : whether to clean text before adding to corpus

    Returns:
        List of strings.

    Usage:
        corpus = build_corpus(train_df, TEXT_COLS)
    """
    if cols is None:
        cols = TEXT_COLS

    corpus = []
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                text = str(row[col])
                corpus.append(clean_text(text) if clean else text)
    return corpus


def build_row_corpus(df: pd.DataFrame,
                     cols: list = None,
                     clean: bool = True) -> list:
    """
    Build a list of strings where each string is the concatenation
    of all specified column values for one row.
    One string per row.

    Usage:
        row_corpus = build_row_corpus(train_df, TEXT_COLS)
        # len(row_corpus) == len(train_df)
    """
    if cols is None:
        cols = TEXT_COLS

    corpus = []
    for _, row in df.iterrows():
        parts    = [str(row[col]) for col in cols if col in df.columns]
        combined = " ".join(parts)
        corpus.append(clean_text(combined) if clean else combined)
    return corpus


def build_token_sentences(df: pd.DataFrame,
                          cols: list = None) -> list:
    """
    Build a list of token lists for Word2Vec training.
    Each element is a list of tokens from one (row, column) pair.

    Usage:
        sentences = build_token_sentences(train_df, TEXT_COLS)
        w2v = Word2Vec(sentences=sentences, ...)
    """
    if cols is None:
        cols = TEXT_COLS

    from gensim.utils import simple_preprocess

    sentences = []
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                tokens = simple_preprocess(str(row[col]))
                if tokens:
                    sentences.append(tokens)
    return sentences