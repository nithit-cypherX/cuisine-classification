"""
vectorstore.py
--------------
Handles all ChromaDB operations.

Responsibilities:
  1. Create and persist a ChromaDB collection
  2. Insert training pool embeddings with metadata
  3. Query for k most similar recipes given a test embedding

IMPORTANT:
  - Only the TRAINING POOL is stored in ChromaDB.
  - The TEST SET is NEVER stored here.
  - Test recipes are only embedded at query time to retrieve similar training examples.

Run this file once after dataset.py to build the vector store:
  python src/vectorstore.py
"""

import chromadb
import pandas as pd
from embeddings import load_model, encode_batch
from config import (
    CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR,
    TRAIN_POOL_CSV, FEW_SHOT_K
)


def build_vectorstore(train_pool: pd.DataFrame) -> chromadb.Collection:
    """
    Embed all training recipes and store them in ChromaDB.

    Args:
        train_pool : DataFrame with columns [id, cuisine, text]

    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"[vectorstore] Deleted existing collection: {CHROMA_COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(CHROMA_COLLECTION_NAME)

    model = load_model()
    texts = train_pool["text"].tolist()
    embeddings = encode_batch(model, texts)

    collection.add(
        ids=[str(i) for i in train_pool["id"].tolist()],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"cuisine": c} for c in train_pool["cuisine"].tolist()]
    )

    print(f"[vectorstore] Stored {len(texts)} recipes in ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    return collection


def load_vectorstore() -> chromadb.Collection:
    """
    Load an existing persisted ChromaDB collection.
    Call this during inference — do NOT rebuild every time.

    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    print(f"[vectorstore] Loaded collection: {CHROMA_COLLECTION_NAME} ({collection.count()} items)")
    return collection


def retrieve_similar(
    collection: chromadb.Collection,
    query_embedding: list[float],
    k: int = FEW_SHOT_K
) -> list[dict]:
    """
    Retrieve k most semantically similar recipes from ChromaDB.
    Called at inference time for each test recipe.

    Args:
        collection      : ChromaDB collection
        query_embedding : Embedding of the test recipe (NOT stored)
        k               : Number of examples to retrieve

    Returns:
        List of dicts with keys: text, cuisine
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": doc, "cuisine": meta["cuisine"]})

    return retrieved


if __name__ == "__main__":
    train_pool = pd.read_csv(TRAIN_POOL_CSV)
    build_vectorstore(train_pool)
