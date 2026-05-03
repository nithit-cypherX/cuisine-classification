"""
vectorstore.py
--------------
Handles all ChromaDB operations for all three embedding types.

Each embedding type gets its OWN separate ChromaDB collection:
  cuisine_recipes_bow     ← BoW vectors
  cuisine_recipes_sparse  ← TF-IDF vectors
  cuisine_recipes_dense   ← SentenceTransformer vectors

This separation ensures retrieval is always done within the same
embedding space — you cannot mix BoW queries against Dense vectors.

IMPORTANT:
  Only the TRAINING POOL is stored in ChromaDB.
  The TEST SET is NEVER stored here.
  Test recipes are only embedded at query time.

Run this file once after dataset.py to build all 3 collections:
  python src/vectorstore.py
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from embeddings import (
    fit_bow, fit_sparse, load_dense_model,
    encode_batch
)
from config import (
    CHROMA_PERSIST_DIR, EMBEDDING_TYPES,
    chroma_collection_name, TRAIN_POOL_CSV
)


def get_chroma_client() -> chromadb.PersistentClient:
    """Create or connect to the local ChromaDB instance."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def build_collection(
    client: chromadb.PersistentClient,
    embedding_type: str,
    texts: list[str],
    ids: list[str],
    cuisines: list[str],
    model_or_vectorizer
) -> chromadb.Collection:
    """
    Build a single ChromaDB collection for one embedding type.

    Args:
        client              : ChromaDB persistent client
        embedding_type      : 'bow', 'sparse', or 'dense'
        texts               : Training ingredient strings
        ids                 : Unique recipe IDs (as strings)
        cuisines            : Cuisine label for each recipe
        model_or_vectorizer : The fitted vectorizer or loaded dense model

    Returns:
        ChromaDB Collection
    """
    collection_name = chroma_collection_name(embedding_type)

    # Delete and recreate if exists (clean rebuild)
    try:
        client.delete_collection(collection_name)
        print(f"[vectorstore] Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"embedding_type": embedding_type}
    )

    # Encode all training texts with the given embedding type
    embeddings = encode_batch(embedding_type, texts, model_or_vectorizer)

    # Insert into ChromaDB in batches to avoid memory issues
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch_end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=[{"cuisine": c} for c in cuisines[i:batch_end]]
        )
        print(f"[vectorstore] [{embedding_type}] Inserted {batch_end}/{len(texts)} recipes...")

    print(f"[vectorstore] ✓ Built collection '{collection_name}' with {len(texts)} recipes\n")
    return collection


def build_all_vectorstores(train_pool: pd.DataFrame) -> None:
    """
    Build all 3 ChromaDB collections (BoW, Sparse, Dense) from the training pool.
    Fits and saves vectorizers for BoW and Sparse.

    Args:
        train_pool : DataFrame with columns [id, cuisine, text]
    """
    client   = get_chroma_client()
    texts    = train_pool["text"].tolist()
    ids      = [str(i) for i in train_pool["id"].tolist()]
    cuisines = train_pool["cuisine"].tolist()

    print("=" * 55)
    print(f"Building {len(EMBEDDING_TYPES)} ChromaDB collections...")
    print(f"Training pool size: {len(texts)} recipes")
    print("=" * 55 + "\n")

    # ── BoW ───────────────────────────────────────────────────────────────────
    print("── [1/3] BoW (Bag of Words) ──────────────────────────────")
    bow_vectorizer = fit_bow(texts)
    build_collection(client, "bow", texts, ids, cuisines, bow_vectorizer)

    # ── Sparse (TF-IDF) ───────────────────────────────────────────────────────
    print("── [2/3] Sparse (TF-IDF) ─────────────────────────────────")
    sparse_vectorizer = fit_sparse(texts)
    build_collection(client, "sparse", texts, ids, cuisines, sparse_vectorizer)

    # ── Dense (SentenceTransformer) ───────────────────────────────────────────
    print("── [3/3] Dense (SentenceTransformer) ─────────────────────")
    dense_model = load_dense_model()
    build_collection(client, "dense", texts, ids, cuisines, dense_model)

    print("=" * 55)
    print("✓ All 3 ChromaDB collections built successfully.")
    print("=" * 55)


def load_collection(embedding_type: str) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection for a given embedding type.
    Called during inference — do NOT rebuild every run.

    Args:
        embedding_type : 'bow', 'sparse', or 'dense'

    Returns:
        ChromaDB Collection
    """
    client = get_chroma_client()
    name   = chroma_collection_name(embedding_type)
    collection = client.get_collection(name)
    print(f"[vectorstore] Loaded '{name}' ({collection.count()} items)")
    return collection


def retrieve_similar(
    collection: chromadb.Collection,
    query_embedding: list[float],
    k: int
) -> list[dict]:
    """
    Retrieve k most semantically similar recipes from ChromaDB.
    Called at inference time for each test recipe.

    Args:
        collection      : ChromaDB collection for the current embedding type
        query_embedding : Embedding of the test recipe (NOT stored)
        k               : Number of examples to retrieve

    Returns:
        List of dicts with keys: text, cuisine
        Returns empty list if k=0 (zero-shot condition)
    """
    if k == 0:
        return []   # Zero-shot: no retrieval, no examples in prompt

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text":    doc,
            "cuisine": meta["cuisine"]
        })

    return retrieved


if __name__ == "__main__":
    train_pool = pd.read_csv(TRAIN_POOL_CSV)
    build_all_vectorstores(train_pool)
