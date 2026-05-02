"""
embeddings.py
-------------
Handles all SentenceTransformer embedding operations.

Responsibilities:
  1. Load the embedding model (downloaded automatically on first run)
  2. Encode a list of text strings into vectors
  3. Encode a single query string at inference time

Model used: all-MiniLM-L6-v2
  - Output dimension : 384
  - Speed            : Fast (CPU-compatible)
  - Quality          : Strong for semantic similarity tasks
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


def load_model() -> SentenceTransformer:
    """Load and return the embedding model."""
    print(f"[embeddings] Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def encode_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Encode a list of texts into embedding vectors.
    Used for encoding the training pool before inserting into ChromaDB.

    Args:
        model : Loaded SentenceTransformer model
        texts : List of ingredient strings

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    print(f"[embeddings] Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def encode_single(model: SentenceTransformer, text: str) -> list[float]:
    """
    Encode a single query text at inference time.
    Used when querying ChromaDB for each test recipe.

    Args:
        model : Loaded SentenceTransformer model
        text  : Single ingredient string

    Returns:
        Single embedding vector as a list of floats
    """
    embedding = model.encode([text], convert_to_numpy=True)
    return embedding[0].tolist()
