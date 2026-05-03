"""
dataset.py
----------
Handles all data loading, filtering, splitting, and formatting.

Responsibilities:
  1. Load train.json from Kaggle
  2. Filter to target cuisines only
  3. Stratified train/test split (80/20)
  4. Sample fixed test set (TEST_SAMPLES_PER_CUISINE per class)
  5. Format ingredient lists into input strings
  6. Save train_pool.csv and test_set.csv

Run this file once before anything else:
  python src/dataset.py
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    TRAIN_JSON_PATH, TRAIN_POOL_CSV, TEST_SET_CSV,
    TARGET_CUISINES, TEST_SAMPLES_PER_CUISINE, RANDOM_SEED,
    DATA_PROCESSED_DIR
)


def load_raw(path: str) -> pd.DataFrame:
    """Load train.json and return as DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"[dataset] Loaded {len(df)} recipes across {df['cuisine'].nunique()} cuisines.")
    return df


def format_ingredients(ingredients: list) -> str:
    """Join ingredient list into a comma-separated string."""
    return ", ".join(ingredients)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter, format, split, and sample the dataset.

    Returns:
        train_pool : DataFrame used for ChromaDB embedding storage
        test_set   : DataFrame used for LLM evaluation (never stored in ChromaDB)
    """
    # Step 1: Filter to target cuisines
    df = df[df["cuisine"].isin(TARGET_CUISINES)].copy()
    print(f"[dataset] After cuisine filter: {len(df)} recipes across {df['cuisine'].nunique()} cuisines.")

    # Step 2: Format input text
    df["text"] = df["ingredients"].apply(format_ingredients)

    # Step 3: Stratified 80/20 split
    train_pool, test_split = train_test_split(
        df,
        test_size=0.2,
        stratify=df["cuisine"],
        random_state=RANDOM_SEED
    )

    # Step 4: Sample fixed number from test split (15 per cuisine)
    test_set = (
        test_split
        .groupby("cuisine")
        .apply(lambda x: x.sample(
            min(len(x), TEST_SAMPLES_PER_CUISINE),
            random_state=RANDOM_SEED
        ))
        .reset_index(drop=True)
    )

    print(f"[dataset] Train pool: {len(train_pool)} recipes")
    print(f"[dataset] Test set:   {len(test_set)} recipes ({TEST_SAMPLES_PER_CUISINE} per cuisine)")
    print(f"[dataset] Test set class distribution:\n{test_set['cuisine'].value_counts()}")

    return train_pool[["id", "cuisine", "text"]], test_set[["id", "cuisine", "text"]]


def save(train_pool: pd.DataFrame, test_set: pd.DataFrame) -> None:
    """Save both splits to the processed data directory."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    train_pool.to_csv(TRAIN_POOL_CSV, index=False)
    test_set.to_csv(TEST_SET_CSV, index=False)
    print(f"[dataset] Saved train pool → {TRAIN_POOL_CSV}")
    print(f"[dataset] Saved test set   → {TEST_SET_CSV}")


if __name__ == "__main__":
    df = load_raw(TRAIN_JSON_PATH)
    train_pool, test_set = prepare(df)
    save(train_pool, test_set)
