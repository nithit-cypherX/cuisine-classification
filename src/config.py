"""
config.py
---------
Central configuration for the entire project.
All constants, paths, and settings are defined here.
Do NOT hardcode values in other files — import from here instead.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Embedding Types ───────────────────────────────────────────────────────────
# Three embedding methods to compare — each gets its own ChromaDB collection
EMBEDDING_TYPES = ["bow", "sparse", "dense"]

# Dense model name (SentenceTransformer)
DENSE_MODEL     = os.getenv("DENSE_MODEL", "all-MiniLM-L6-v2")

# BoW / Sparse vocab size — limits vector dimension to keep ChromaDB manageable
BOW_MAX_FEATURES    = 5000   # CountVectorizer max vocabulary size
SPARSE_MAX_FEATURES = 5000   # TfidfVectorizer max vocabulary size

# ── ChromaDB ──────────────────────────────────────────────────────────────────
# Each embedding type gets its own collection so they never mix
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def chroma_collection_name(embedding_type: str) -> str:
    """Return collection name for a given embedding type."""
    return f"cuisine_recipes_{embedding_type}"
    # Results in: cuisine_recipes_bow, cuisine_recipes_sparse, cuisine_recipes_dense

# ── Experiment ────────────────────────────────────────────────────────────────
TARGET_CUISINES = os.getenv(
    "TARGET_CUISINES",
    "italian,mexican,indian,thai,japanese,southern_us"
).split(",")

TEST_SAMPLES_PER_CUISINE = int(os.getenv("TEST_SAMPLES_PER_CUISINE", 15))
RANDOM_SEED              = int(os.getenv("RANDOM_SEED", 42))

# k values to experiment with:
#   k=0  → Zero-shot (no examples retrieved, prompt has no context)
#   k=4  → Dynamic few-shot with 4 retrieved examples
#   k=8  → Dynamic few-shot with 8 retrieved examples
#   k=16 → Dynamic few-shot with 16 retrieved examples
K_VALUES = [0, 4, 8, 16]

# LLM settings — do NOT change mid-experiment
LLM_TEMPERATURE = 0     # Always 0 for deterministic, reproducible outputs
LLM_MAX_TOKENS  = 20    # Label only — keep short

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR        = "./data/raw"
DATA_PROCESSED_DIR  = "./data/processed"
RESULTS_PRED_DIR    = "./results/predictions"
RESULTS_FIG_DIR     = "./results/figures"

TRAIN_JSON_PATH     = f"{DATA_RAW_DIR}/train.json"
TRAIN_POOL_CSV      = f"{DATA_PROCESSED_DIR}/train_pool.csv"
TEST_SET_CSV        = f"{DATA_PROCESSED_DIR}/test_set.csv"
PREDICTIONS_CSV     = f"{RESULTS_PRED_DIR}/predictions.csv"

# Fitted vectorizer save paths (BoW and Sparse must be fitted on train, reused on test)
BOW_VECTORIZER_PATH    = f"{DATA_PROCESSED_DIR}/bow_vectorizer.pkl"
SPARSE_VECTORIZER_PATH = f"{DATA_PROCESSED_DIR}/sparse_vectorizer.pkl"
