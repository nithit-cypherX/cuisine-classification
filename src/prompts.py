"""
prompts.py
----------
Defines all LLM prompt templates used in the experiment.

build_prompt() is the single entry point used by inference.py.
It automatically produces:
  - Zero-shot prompt  when retrieved_examples is empty (k=0)
  - Dynamic few-shot  when retrieved_examples has content (k>0)

DO NOT modify prompts mid-experiment.
If testing a new prompt style, add a new function — never overwrite.
"""

from config import TARGET_CUISINES

# Label list for parsing outputs — must match exactly what is stored in the dataset
LABEL_LIST = TARGET_CUISINES

# Human-readable label string shown inside the prompt
CUISINE_OPTIONS = ", ".join([c.replace("_", " ").title() for c in TARGET_CUISINES])
# → Italian, Mexican, Indian, Thai, Japanese, Southern Us


def build_prompt(text: str, retrieved_examples: list[dict]) -> str:
    """
    Build the LLM prompt for a single test recipe.

    Automatically switches between zero-shot and dynamic few-shot
    based on whether retrieved_examples is empty or not.

    Args:
        text               : Test recipe ingredient string
        retrieved_examples : List of dicts with keys 'text' and 'cuisine'
                             Empty list → zero-shot
                             Non-empty  → dynamic few-shot

    Returns:
        Fully formatted prompt string ready to send to the OpenAI API
    """
    if not retrieved_examples:
        return _zero_shot_prompt(text)
    else:
        return _dynamic_few_shot_prompt(text, retrieved_examples)


def _zero_shot_prompt(text: str) -> str:
    """
    Zero-shot prompt: no examples, just the ingredient list and label options.
    Used when k=0.

    Args:
        text : Ingredient string

    Returns:
        Formatted prompt string
    """
    return f"""Classify the following recipe into exactly one cuisine category.
Choose only from: {CUISINE_OPTIONS}.
Reply with the cuisine name only — no explanation, no extra text.

Recipe ingredients: {text}
Cuisine:"""


def _dynamic_few_shot_prompt(text: str, retrieved_examples: list[dict]) -> str:
    """
    Dynamic few-shot prompt: uses examples retrieved from ChromaDB as context.
    The examples are different for every test query — hence "dynamic".
    Used when k > 0.

    Args:
        text               : Test recipe ingredient string
        retrieved_examples : List of dicts {'text': ..., 'cuisine': ...}

    Returns:
        Formatted prompt string
    """
    examples_block = ""
    for ex in retrieved_examples:
        label = ex["cuisine"].replace("_", " ").title()
        examples_block += f"Ingredients: {ex['text']}\nCuisine: {label}\n\n"

    return f"""Classify the following recipe into exactly one cuisine category.
Choose only from: {CUISINE_OPTIONS}.
Reply with the cuisine name only — no explanation, no extra text.

Here are some reference examples:

{examples_block.strip()}

Now classify this recipe:
Ingredients: {text}
Cuisine:"""
