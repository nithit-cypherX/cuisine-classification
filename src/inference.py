"""
inference.py
------------
Runs the full experiment across all embedding types and k values.

Experiment matrix:
  Embedding types : bow, sparse, dense         (3 types)
  k values        : 0, 4, 8, 16               (4 values)
  Total conditions: 3 × 4 = 12 conditions

For each condition:
  k=0  → Zero-shot: test recipe + no examples → OpenAI API
  k>0  → Dynamic few-shot:
          embed test recipe → query ChromaDB (k examples) → OpenAI API

All 12 conditions are run on the SAME 90 test recipes.
Results are saved as one CSV with one column per condition.

Run after dataset.py and vectorstore.py:
  python src/inference.py

Output: results/predictions/predictions.csv
"""

import os
import time
import pandas as pd
from openai import OpenAI

from embeddings import (
    load_bow, load_sparse, load_dense_model,
    encode_single
)
from vectorstore import load_collection, retrieve_similar
from prompts import build_prompt, LABEL_LIST
from config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    TEST_SET_CSV, PREDICTIONS_CSV,
    RESULTS_PRED_DIR,
    EMBEDDING_TYPES, K_VALUES
)


# ── Initialise OpenAI client ──────────────────────────────────────────────────
llm_client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm(prompt: str) -> str:
    """Send a prompt to OpenAI and return the raw text response."""
    response = llm_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return response.choices[0].message.content.strip()


def parse_label(raw_output: str) -> str:
    """
    Extract a clean cuisine label from the LLM's raw output.
    Returns 'invalid' if no known label is found.

    Args:
        raw_output : Raw string returned by the LLM

    Returns:
        Matched cuisine label (e.g. 'italian') or 'invalid'
    """
    text = raw_output.lower()
    for label in LABEL_LIST:
        if label in text or label.replace("_", " ") in text:
            return label
    return "invalid"


def load_all_models() -> dict:
    """
    Load all vectorizers and models needed for inference.
    Loads once at startup — not repeatedly per recipe.

    Returns:
        Dict keyed by embedding type: {'bow': ..., 'sparse': ..., 'dense': ...}
    """
    print("[inference] Loading all embedding models and vectorizers...")
    return {
        "bow":    load_bow(),
        "sparse": load_sparse(),
        "dense":  load_dense_model()
    }


def load_all_collections() -> dict:
    """
    Load all ChromaDB collections needed for inference.

    Returns:
        Dict keyed by embedding type: {'bow': collection, ...}
    """
    print("[inference] Loading all ChromaDB collections...")
    return {
        emb_type: load_collection(emb_type)
        for emb_type in EMBEDDING_TYPES
    }


def run_experiment(test_set: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 12 conditions (3 embedding types × 4 k values) on the test set.

    For each test recipe and each condition:
      1. Embed the test recipe using the current embedding type
      2. If k > 0: query ChromaDB to retrieve k similar training recipes
      3. Build the prompt (zero-shot or dynamic few-shot)
      4. Call the OpenAI API
      5. Parse the output label

    Args:
        test_set : DataFrame with columns [id, cuisine, text]

    Returns:
        DataFrame with original columns + one column per condition
    """
    models      = load_all_models()
    collections = load_all_collections()

    results_df = test_set.copy()
    total      = len(test_set)

    for emb_type in EMBEDDING_TYPES:
        for k in K_VALUES:
            condition = f"{emb_type}_k{k}"
            raw_col   = f"{condition}_raw"
            pred_col  = f"{condition}_pred"

            print(f"\n{'='*55}")
            print(f"Condition: embedding={emb_type.upper()}, k={k}")
            print(f"{'='*55}")

            raw_outputs = []
            predictions = []

            for idx, row in test_set.iterrows():
                print(f"  [{idx+1}/{total}] {row['cuisine']} | {emb_type} | k={k}")

                # Step 1: Embed the test recipe
                query_embedding = encode_single(emb_type, row["text"], models[emb_type])

                # Step 2: Retrieve k examples from ChromaDB (empty list if k=0)
                retrieved = retrieve_similar(collections[emb_type], query_embedding, k=k)

                # Step 3: Build prompt
                prompt = build_prompt(row["text"], retrieved)

                # Step 4: Call LLM
                raw = call_llm(prompt)
                raw_outputs.append(raw)

                # Step 5: Parse output
                pred = parse_label(raw)
                predictions.append(pred)

                time.sleep(0.3)   # Avoid OpenAI rate limiting

            results_df[raw_col]  = raw_outputs
            results_df[pred_col] = predictions

            # Per-condition summary
            invalid_n    = predictions.count("invalid")
            invalid_rate = invalid_n / total
            print(f"\n  ✓ Done | Invalid outputs: {invalid_n}/{total} ({invalid_rate:.1%})")

    return results_df


def save_predictions(df: pd.DataFrame) -> None:
    """Save all prediction results to CSV."""
    os.makedirs(RESULTS_PRED_DIR, exist_ok=True)
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\n[inference] All predictions saved → {PREDICTIONS_CSV}")


if __name__ == "__main__":
    test_set = pd.read_csv(TEST_SET_CSV)
    print(f"[inference] Test set: {len(test_set)} recipes")
    print(f"[inference] Conditions: {len(EMBEDDING_TYPES)} embedding types × {len(K_VALUES)} k values = {len(EMBEDDING_TYPES)*len(K_VALUES)} total\n")

    results = run_experiment(test_set)
    save_predictions(results)

    # Final summary table
    print("\n── Final Summary ────────────────────────────────────────")
    for emb_type in EMBEDDING_TYPES:
        for k in K_VALUES:
            col          = f"{emb_type}_k{k}_pred"
            invalid_rate = (results[col] == "invalid").mean()
            print(f"  {emb_type:8s} k={k:2d} | invalid rate: {invalid_rate:.1%}")
