# embeddings.py
# TF-IDF and Word2Vec embedding generation and utilities

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition           import PCA, TruncatedSVD

from gensim.models  import Word2Vec
from gensim.utils   import simple_preprocess

from config.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    TFIDF_MAX_DF, TFIDF_SUBLINEAR_TF,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT,
    W2V_SG, W2V_EPOCHS, W2V_WORKERS, W2V_SEED,
    OPTION_COLS, PROMPT_COL, MODEL_DIR
)
from .preprocessing import (
    clean_text, build_row_corpus, build_token_sentences
)


# ------------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------------

class TFIDFEmbedder:
    """
    Wrapper around sklearn TfidfVectorizer for the MCQ pipeline.

    Usage:
        embedder = TFIDFEmbedder()
        embedder.fit(corpus)                     # corpus = list of strings
        matrix  = embedder.transform(corpus)    # scipy sparse matrix
        vec     = embedder.transform_one("some text")
    """

    def __init__(self,
                 max_features: int  = TFIDF_MAX_FEATURES,
                 ngram_range: tuple = TFIDF_NGRAM_RANGE,
                 min_df: int        = TFIDF_MIN_DF,
                 max_df: float      = TFIDF_MAX_DF,
                 sublinear_tf: bool = TFIDF_SUBLINEAR_TF):

        self.vectorizer = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            min_df       = min_df,
            max_df       = max_df,
            sublinear_tf = sublinear_tf
        )
        self.is_fitted = False

    def fit(self, corpus: list) -> "TFIDFEmbedder":
        """
        Fit the TF-IDF vocabulary on a list of text strings.

        Usage:
            embedder.fit(corpus)
        """
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        print(f"TF-IDF fitted  | vocab size : {len(self.vectorizer.vocabulary_)}")
        return self

    def transform(self, corpus: list):
        """
        Transform a list of strings into a sparse TF-IDF matrix.

        Usage:
            matrix = embedder.transform(corpus)
            # shape: (n_docs, max_features)
        """
        self._check_fitted()
        return self.vectorizer.transform(corpus)

    def fit_transform(self, corpus: list):
        """Fit and transform in one step."""
        self.fit(corpus)
        return self.transform(corpus)

    def transform_one(self, text: str):
        """
        Transform a single string into a sparse TF-IDF vector.

        Usage:
            vec = embedder.transform_one("What is fusion?")
        """
        self._check_fitted()
        cleaned = clean_text(text)
        return self.vectorizer.transform([cleaned])

    def get_feature_names(self) -> np.ndarray:
        """Return feature (token) names."""
        self._check_fitted()
        return self.vectorizer.get_feature_names_out()

    def top_features(self, n: int = 20) -> pd.DataFrame:
        """
        Return the n features with the highest mean TF-IDF score
        across the fitted corpus.

        Usage:
            top_df = embedder.top_features(20)
        """
        self._check_fitted()
        # Requires re-transforming the training corpus, store mean
        # during fit_transform if needed.
        raise NotImplementedError(
            "Call top_features_from_matrix(matrix, n) instead."
        )

    def save(self, path: str) -> None:
        """
        Save the fitted vectorizer to disk.

        Usage:
            embedder.save("models/tfidf.pkl")
        """
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"TF-IDF saved to {path}")

    def load(self, path: str) -> "TFIDFEmbedder":
        """
        Load a previously saved vectorizer from disk.

        Usage:
            embedder = TFIDFEmbedder().load("models/tfidf.pkl")
        """
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"TF-IDF loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("TFIDFEmbedder is not fitted yet. Call .fit() first.")


def top_features_from_matrix(matrix, feature_names: np.ndarray,
                              n: int = 20) -> pd.DataFrame:
    """
    Compute the top-n features by mean TF-IDF score across all documents.

    Args:
        matrix        : sparse TF-IDF matrix (n_docs x vocab)
        feature_names : array of feature names from TFIDFEmbedder.get_feature_names()
        n             : number of top features to return

    Returns:
        DataFrame with columns ['feature', 'mean_tfidf'].

    Usage:
        top_df = top_features_from_matrix(matrix, embedder.get_feature_names(), 20)
    """
    mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[-n:][::-1]
    return pd.DataFrame({
        "feature"    : feature_names[top_indices],
        "mean_tfidf" : mean_scores[top_indices]
    })


# ------------------------------------------------------------------
# WORD2VEC
# ------------------------------------------------------------------

class Word2VecEmbedder:
    """
    Wrapper around Gensim Word2Vec for the MCQ pipeline.

    Usage:
        embedder = Word2VecEmbedder()
        embedder.fit(sentences)               # sentences = list of token lists
        vec = embedder.get_word_vector("time")
        doc_vec = embedder.get_doc_vector("Martin Heidegger's view on time")
    """

    def __init__(self,
                 vector_size: int = W2V_VECTOR_SIZE,
                 window: int      = W2V_WINDOW,
                 min_count: int   = W2V_MIN_COUNT,
                 sg: int          = W2V_SG,
                 epochs: int      = W2V_EPOCHS,
                 workers: int     = W2V_WORKERS,
                 seed: int        = W2V_SEED):

        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.sg          = sg
        self.epochs      = epochs
        self.workers     = workers
        self.seed        = seed
        self.model       = None

    def fit(self, sentences: list) -> "Word2VecEmbedder":
        """
        Train Word2Vec on a list of token lists.

        Args:
            sentences : list of lists of string tokens

        Usage:
            sentences = build_token_sentences(train_df, TEXT_COLS)
            embedder.fit(sentences)
        """
        self.model = Word2Vec(
            sentences   = sentences,
            vector_size = self.vector_size,
            window      = self.window,
            min_count   = self.min_count,
            sg          = self.sg,
            epochs      = self.epochs,
            workers     = self.workers,
            seed        = self.seed
        )
        print(f"Word2Vec trained | vocab size : {len(self.model.wv.key_to_index)}")
        return self

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Return the embedding vector for a single word.
        Returns a zero vector if the word is not in the vocabulary.

        Usage:
            vec = embedder.get_word_vector("fusion")
        """
        self._check_fitted()
        if word in self.model.wv:
            return self.model.wv[word]
        return np.zeros(self.vector_size)

    def get_doc_vector(self, text: str) -> np.ndarray:
        """
        Compute a document embedding by mean-pooling word vectors.
        Words absent from the vocabulary are ignored.
        Returns a zero vector if no word is in the vocabulary.

        Usage:
            doc_vec = embedder.get_doc_vector("What is accelerator based fusion?")
        """
        self._check_fitted()
        tokens  = simple_preprocess(str(text))
        vectors = [
            self.model.wv[t] for t in tokens if t in self.model.wv
        ]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    def get_doc_vectors_batch(self, texts: list) -> np.ndarray:
        """
        Compute document embeddings for a list of strings.

        Returns:
            np.ndarray of shape (len(texts), vector_size)

        Usage:
            vecs = embedder.get_doc_vectors_batch(train_df["prompt"].tolist())
        """
        return np.vstack([self.get_doc_vector(t) for t in texts])

    def most_similar(self, word: str, topn: int = 5) -> list:
        """
        Return the most similar words to a given word.

        Usage:
            similar = embedder.most_similar("fusion", topn=5)
        """
        self._check_fitted()
        if word not in self.model.wv:
            print(f"Word '{word}' not in vocabulary.")
            return []
        return self.model.wv.most_similar(word, topn=topn)

    def save(self, path: str) -> None:
        """
        Save the trained Word2Vec model to disk.

        Usage:
            embedder.save("models/w2v.model")
        """
        self._check_fitted()
        self.model.save(path)
        print(f"Word2Vec saved to {path}")

    def load(self, path: str) -> "Word2VecEmbedder":
        """
        Load a previously saved Word2Vec model from disk.

        Usage:
            embedder = Word2VecEmbedder().load("models/w2v.model")
        """
        self.model = Word2Vec.load(path)
        print(f"Word2Vec loaded from {path}")
        return self

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Word2VecEmbedder is not fitted yet. Call .fit() first."
            )


# ------------------------------------------------------------------
# DIMENSIONALITY REDUCTION
# ------------------------------------------------------------------

def reduce_with_pca(matrix: np.ndarray,
                    n_components: int = 2,
                    seed: int         = 42) -> tuple:
    """
    Reduce a dense embedding matrix to n_components dimensions using PCA.

    Returns:
        (reduced_matrix, pca_object)

    Usage:
        reduced, pca = reduce_with_pca(word_vectors, n_components=2)
    """
    pca     = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(matrix)
    print(f"PCA | explained variance : "
          f"{pca.explained_variance_ratio_.sum()*100:.2f}%")
    return reduced, pca


def reduce_with_svd(sparse_matrix,
                    n_components: int = 2,
                    seed: int         = 42) -> tuple:
    """
    Reduce a sparse TF-IDF matrix using TruncatedSVD (LSA).

    Returns:
        (reduced_matrix, svd_object)

    Usage:
        reduced, svd = reduce_with_svd(tfidf_matrix, n_components=2)
    """
    svd     = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(sparse_matrix)
    print(f"SVD | explained variance : "
          f"{svd.explained_variance_ratio_.sum()*100:.2f}%")
    return reduced, svd