"""
inference.py
------------
Runs the full experiment: applies both prompt strategies to the test set
and saves all predictions to a CSV.

Pipeline per test recipe:
  Zero-shot        → format prompt → call OpenAI API → parse output
  Dynamic few-shot → embed recipe → query ChromaDB → format prompt
                   → call OpenAI API → parse output

Run after dataset.py and vectorstore.py have been completed:
  python src/inference.py

Output: results/predictions/predictions.csv
  Columns: id, true_label, zero_shot_pred, dynamic_few_shot_pred,
           zero_shot_raw, dynamic_few_shot_raw
"""

import os
import time
import pandas as pd
from openai import OpenAI
from embeddings import load_model, encode_single
from vectorstore import load_vectorstore, retrieve_similar
from prompts import zero_shot, dynamic_few_shot, LABEL_LIST
from config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    TEST_SET_CSV, PREDICTIONS_CSV,
    RESULTS_PRED_DIR, FEW_SHOT_K
)


# ── Initialise clients ────────────────────────────────────────────────────────
client     = OpenAI(api_key=OPENAI_API_KEY)
embed_model = load_model()
collection  = load_vectorstore()


def call_llm(prompt: str) -> str:
    """
    Send a prompt to the OpenAI API and return the raw text response.

    Args:
        prompt : Fully formatted prompt string

    Returns:
        Raw string response from the LLM
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return response.choices[0].message.content.strip()


def parse_label(raw_output: str) -> str:
    """
    Extract a clean cuisine label from the LLM's raw output.
    Returns 'invalid' if no known cuisine name is found.

    Args:
        raw_output : Raw string returned by the LLM

    Returns:
        Matched cuisine label (e.g. 'italian') or 'invalid'
    """
    normalised = raw_output.lower().replace(" ", "_")
    for label in LABEL_LIST:
        if label in normalised or label.replace("_", " ") in raw_output.lower():
            return label
    return "invalid"


def run_experiment(test_set: pd.DataFrame) -> pd.DataFrame:
    """
    Run both prompt strategies on the entire test set.

    Args:
        test_set : DataFrame with columns [id, cuisine, text]

    Returns:
        DataFrame with prediction columns added
    """
    zero_shot_raws    = []
    zero_shot_preds   = []
    few_shot_raws     = []
    few_shot_preds    = []

    total = len(test_set)
    for i, row in test_set.iterrows():
        print(f"[inference] Processing {i+1}/{total} — {row['cuisine']}")

        # ── Strategy 1: Zero-shot ─────────────────────────────────────────
        zs_prompt = zero_shot(row["text"])
        zs_raw    = call_llm(zs_prompt)
        zs_pred   = parse_label(zs_raw)
        zero_shot_raws.append(zs_raw)
        zero_shot_preds.append(zs_pred)

        # ── Strategy 2: Dynamic few-shot ──────────────────────────────────
        query_embedding = encode_single(embed_model, row["text"])
        retrieved       = retrieve_similar(collection, query_embedding, k=FEW_SHOT_K)
        fs_prompt       = dynamic_few_shot(row["text"], retrieved)
        fs_raw          = call_llm(fs_prompt)
        fs_pred         = parse_label(fs_raw)
        few_shot_raws.append(fs_raw)
        few_shot_preds.append(fs_pred)

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    test_set = test_set.copy()
    test_set["zero_shot_raw"]            = zero_shot_raws
    test_set["zero_shot_pred"]           = zero_shot_preds
    test_set["dynamic_few_shot_raw"]     = few_shot_raws
    test_set["dynamic_few_shot_pred"]    = few_shot_preds

    return test_set


def save_predictions(df: pd.DataFrame) -> None:
    """Save prediction results to CSV."""
    os.makedirs(RESULTS_PRED_DIR, exist_ok=True)
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[inference] Predictions saved → {PREDICTIONS_CSV}")


if __name__ == "__main__":
    test_set = pd.read_csv(TEST_SET_CSV)
    print(f"[inference] Running experiment on {len(test_set)} test recipes...")
    results = run_experiment(test_set)
    save_predictions(results)

    # Quick summary
    for col in ["zero_shot_pred", "dynamic_few_shot_pred"]:
        invalid_rate = (results[col] == "invalid").mean()
        print(f"[inference] {col} invalid rate: {invalid_rate:.1%}")
