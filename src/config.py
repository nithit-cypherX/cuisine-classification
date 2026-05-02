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

# ── API ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "cuisine_recipes")
CHROMA_PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ── Experiment ───────────────────────────────────────────────────────────────
TARGET_CUISINES = os.getenv(
    "TARGET_CUISINES",
    "italian,mexican,indian,thai,japanese,southern_us"
).split(",")

TEST_SAMPLES_PER_CUISINE = int(os.getenv("TEST_SAMPLES_PER_CUISINE", 15))
FEW_SHOT_K               = int(os.getenv("FEW_SHOT_K", 3))
RANDOM_SEED              = int(os.getenv("RANDOM_SEED", 42))
LLM_TEMPERATURE          = 0       # Always 0 for reproducibility
LLM_MAX_TOKENS           = 20      # Label only — keep short

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR        = "./data/raw"
DATA_PROCESSED_DIR  = "./data/processed"
DATA_RESULTS_DIR    = "./data/results"
RESULTS_PRED_DIR    = "./results/predictions"
RESULTS_FIG_DIR     = "./results/figures"

TRAIN_JSON_PATH     = f"{DATA_RAW_DIR}/train.json"
TRAIN_POOL_CSV      = f"{DATA_PROCESSED_DIR}/train_pool.csv"
TEST_SET_CSV        = f"{DATA_PROCESSED_DIR}/test_set.csv"
PREDICTIONS_CSV     = f"{RESULTS_PRED_DIR}/predictions.csv"
