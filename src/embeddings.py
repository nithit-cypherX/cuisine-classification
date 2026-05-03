"""
embeddings.py
-------------
Handles all embedding operations for all three methods.

Three embedding types supported:
  1. BoW    — Bag of Words using CountVectorizer (word count vectors)
  2. Sparse — TF-IDF using TfidfVectorizer (weighted sparse vectors)
  3. Dense  — SentenceTransformer all-MiniLM-L6-v2 (semantic vectors)

Key rule:
  BoW and Sparse vectorizers MUST be fitted on the training pool only,
  then saved to disk and reused for test queries.
  Dense model encodes independently — no fitting needed.

Output dimension:
  BoW    → BOW_MAX_FEATURES    (default 5000)
  Sparse → SPARSE_MAX_FEATURES (default 5000)
  Dense  → 384 (fixed by model)
"""

import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from config import (
    DENSE_MODEL,
    BOW_MAX_FEATURES, SPARSE_MAX_FEATURES,
    BOW_VECTORIZER_PATH, SPARSE_VECTORIZER_PATH,
    DATA_PROCESSED_DIR
)


# ── BoW (Bag of Words) ────────────────────────────────────────────────────────

def fit_bow(texts: list[str]) -> CountVectorizer:
    """
    Fit a CountVectorizer on training texts and save to disk.
    Call this ONCE during vectorstore build — never on test data.

    Args:
        texts : List of training ingredient strings

    Returns:
        Fitted CountVectorizer
    """
    print(f"[embeddings] Fitting BoW vectorizer on {len(texts)} texts...")
    vectorizer = CountVectorizer(max_features=BOW_MAX_FEATURES)
    vectorizer.fit(texts)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(BOW_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[embeddings] BoW vectorizer saved → {BOW_VECTORIZER_PATH}")
    return vectorizer


def load_bow() -> CountVectorizer:
    """Load a previously fitted CountVectorizer from disk."""
    with open(BOW_VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def encode_bow_batch(texts: list[str], vectorizer: CountVectorizer) -> list[list[float]]:
    """
    Encode a list of texts using the fitted BoW vectorizer.
    Converts sparse matrix to dense for ChromaDB compatibility.

    Args:
        texts      : List of ingredient strings
        vectorizer : Fitted CountVectorizer

    Returns:
        List of dense vectors (list of floats)
    """
    sparse_matrix = vectorizer.transform(texts)
    return sparse_matrix.toarray().tolist()


def encode_bow_single(text: str, vectorizer: CountVectorizer) -> list[float]:
    """Encode a single query text using the fitted BoW vectorizer."""
    sparse_matrix = vectorizer.transform([text])
    return sparse_matrix.toarray()[0].tolist()


# ── Sparse (TF-IDF) ───────────────────────────────────────────────────────────

def fit_sparse(texts: list[str]) -> TfidfVectorizer:
    """
    Fit a TfidfVectorizer on training texts and save to disk.
    Call this ONCE during vectorstore build — never on test data.

    TF-IDF weights rare but distinctive ingredients higher than common ones.
    This makes it better than BoW for distinguishing similar cuisines.

    Args:
        texts : List of training ingredient strings

    Returns:
        Fitted TfidfVectorizer
    """
    print(f"[embeddings] Fitting Sparse (TF-IDF) vectorizer on {len(texts)} texts...")
    vectorizer = TfidfVectorizer(max_features=SPARSE_MAX_FEATURES)
    vectorizer.fit(texts)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(SPARSE_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[embeddings] Sparse vectorizer saved → {SPARSE_VECTORIZER_PATH}")
    return vectorizer


def load_sparse() -> TfidfVectorizer:
    """Load a previously fitted TfidfVectorizer from disk."""
    with open(SPARSE_VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def encode_sparse_batch(texts: list[str], vectorizer: TfidfVectorizer) -> list[list[float]]:
    """
    Encode a list of texts using the fitted TF-IDF vectorizer.
    Converts sparse matrix to dense for ChromaDB compatibility.

    Args:
        texts      : List of ingredient strings
        vectorizer : Fitted TfidfVectorizer

    Returns:
        List of dense vectors (list of floats)
    """
    sparse_matrix = vectorizer.transform(texts)
    return sparse_matrix.toarray().tolist()


def encode_sparse_single(text: str, vectorizer: TfidfVectorizer) -> list[float]:
    """Encode a single query text using the fitted TF-IDF vectorizer."""
    sparse_matrix = vectorizer.transform([text])
    return sparse_matrix.toarray()[0].tolist()


# ── Dense (SentenceTransformer) ───────────────────────────────────────────────

def load_dense_model() -> SentenceTransformer:
    """Load the SentenceTransformer model. Downloads on first run."""
    print(f"[embeddings] Loading Dense model: {DENSE_MODEL}")
    return SentenceTransformer(DENSE_MODEL)


def encode_dense_batch(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """
    Encode a list of texts using SentenceTransformer.

    Args:
        texts : List of ingredient strings
        model : Loaded SentenceTransformer model

    Returns:
        List of 384-dimensional vectors
    """
    print(f"[embeddings] Dense encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def encode_dense_single(text: str, model: SentenceTransformer) -> list[float]:
    """Encode a single query text using the Dense model."""
    embedding = model.encode([text], convert_to_numpy=True)
    return embedding[0].tolist()


# ── Unified interface ─────────────────────────────────────────────────────────

def encode_batch(embedding_type: str, texts: list[str], model_or_vectorizer) -> list[list[float]]:
    """
    Unified batch encoder — routes to correct method based on embedding_type.

    Args:
        embedding_type      : 'bow', 'sparse', or 'dense'
        texts               : List of ingredient strings
        model_or_vectorizer : CountVectorizer / TfidfVectorizer / SentenceTransformer

    Returns:
        List of embedding vectors
    """
    if embedding_type == "bow":
        return encode_bow_batch(texts, model_or_vectorizer)
    elif embedding_type == "sparse":
        return encode_sparse_batch(texts, model_or_vectorizer)
    elif embedding_type == "dense":
        return encode_dense_batch(texts, model_or_vectorizer)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}. Choose from: bow, sparse, dense")


def encode_single(embedding_type: str, text: str, model_or_vectorizer) -> list[float]:
    """
    Unified single encoder — routes to correct method based on embedding_type.

    Args:
        embedding_type      : 'bow', 'sparse', or 'dense'
        text                : Single ingredient string
        model_or_vectorizer : CountVectorizer / TfidfVectorizer / SentenceTransformer

    Returns:
        Single embedding vector
    """
    if embedding_type == "bow":
        return encode_bow_single(text, model_or_vectorizer)
    elif embedding_type == "sparse":
        return encode_sparse_single(text, model_or_vectorizer)
    elif embedding_type == "dense":
        return encode_dense_single(text, model_or_vectorizer)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}. Choose from: bow, sparse, dense")
