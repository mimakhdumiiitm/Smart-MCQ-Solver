# preprocessing.py
# All text cleaning, tokenization, and feature engineering functions

import re
import pandas as pd

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from config.config import (
    TEXT_COLS,REMOVE_STOPWORDS, LEMMATIZE, STEM, MIN_TOKEN_LENGTH
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

    return series.apply(
        lambda x: clean_text(x, remove_stopwords, lemmatize, stem)
    )


def clean_dataframe(df: pd.DataFrame,
                    cols: list         = None,
                    suffix: str        = "_clean",
                    remove_stopwords: bool = REMOVE_STOPWORDS,
                    lemmatize: bool        = LEMMATIZE,
                    stem: bool             = STEM) -> pd.DataFrame:

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