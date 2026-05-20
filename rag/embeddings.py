"""
Embeddings module using TF-IDF (local, no external API needed).
Works offline with zero model downloads — uses sklearn.
For production deployments with internet access, swap embed_texts/embed_query
to use sentence-transformers or Hugging Face inference endpoints.
"""
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "../database/tfidf_vectorizer.pkl")

_vectorizer = None


def get_vectorizer() -> TfidfVectorizer:
    global _vectorizer
    if _vectorizer is None and os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
    return _vectorizer


def fit_and_embed(texts: list) -> np.ndarray:
    """
    Fit TF-IDF on the full corpus and return dense embeddings.
    Saves the vectorizer to disk for later query embedding.
    """
    global _vectorizer
    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)

    _vectorizer = TfidfVectorizer(
        max_features=2048,
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=1,
    )

    matrix = _vectorizer.fit_transform(texts)
    dense = matrix.toarray().astype("float32")
    dense = normalize(dense, norm='l2', axis=1)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(_vectorizer, f)

    print(f"TF-IDF fitted: {len(texts)} docs, vocab={len(_vectorizer.vocabulary_)}, dim={dense.shape[1]}")
    return dense


def embed_texts(texts: list) -> np.ndarray:
    """Embed texts using the existing TF-IDF vectorizer or fit a new one if needed."""
    vec = get_vectorizer()
    if vec is None:
        return fit_and_embed(texts)

    matrix = vec.transform(texts).toarray().astype("float32")
    return normalize(matrix, norm='l2', axis=1)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, returns (1, D) array."""
    vec = get_vectorizer()
    if vec is None:
        raise RuntimeError("Vectorizer not fitted. Run build_index first.")
    matrix = vec.transform([query]).toarray().astype("float32")
    return normalize(matrix, norm='l2', axis=1)
